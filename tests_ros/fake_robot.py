"""An in-process fake FR3 cell made of real rclpy endpoints, for integration tests.

It runs in its own ``rclpy.Context`` (a separate DDS participant from the backend under test)
and provides what a franka_ros2 + MoveIt 2 bringup would:

- ``/joint_states`` at 100 Hz (arm + finger joints) and dynamic tf ``fr3_link0 -> fr3_hand_tcp``
  (FR3 kinematics), static tf ``world -> fr3_link0``
- ``/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame`` (best-effort QoS, like franka_ros2)
- ``/fr3_arm_controller/follow_joint_trajectory`` that really interpolates the trajectory in
  time, publishes feedback and honours cancel requests
- ``/controller_manager/list_controllers`` and ``/switch_controller``
- MoveIt-like ``/compute_fk``, ``/plan_kinematic_path``, ``/compute_cartesian_path`` backed by
  armguard's FR3 kinematics (they are NOT MoveIt: no collision checking, simple timing)
- franka_gripper ``/franka_gripper/{move,grasp,homing}`` (if franka_msgs is importable) and
  ``/franka_gripper/gripper_action`` (control_msgs/GripperCommand), ``/franka_gripper/joint_states``
- franka_hardware ``/service_server/set_force_torque_collision_behavior`` and
  ``/action_server/error_recovery`` (if franka_msgs is importable)
- camera topics: raw ``/camera/color/image_raw`` (rgb8 640x480) and
  ``/camera/color/image_raw/compressed`` (PNG)
"""

from __future__ import annotations

import contextlib
import itertools
import math
import threading
import time
from collections.abc import Sequence
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped, TransformStamped, WrenchStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan, GetPositionFK
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import JointState as JointStateMsg
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from trajectory_msgs.msg import JointTrajectoryPoint

from armguard_mcp import geometry as g
from armguard_mcp.backends.fake import FR3_MAX_ACCELERATION, FakeBackend
from armguard_mcp.imaging import gradient_png
from armguard_mcp.kinematics import FR3_JOINT_NAMES, FR3_READY, FR3Kinematics

try:
    from franka_msgs.action import ErrorRecovery, Grasp, Homing, Move
    from franka_msgs.srv import SetForceTorqueCollisionBehavior

    HAVE_FRANKA_MSGS = True
except ImportError:  # franka_msgs is built from source (franka_ros2); optional
    HAVE_FRANKA_MSGS = False

MAX_VEL = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)
WRENCH_TOPIC = "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame"


def _dur(t: float) -> Duration:
    sec = math.floor(t)
    return Duration(sec=int(sec), nanosec=round((t - sec) * 1e9) % 1_000_000_000)


def _s(d: Any) -> float:
    return d.sec + d.nanosec * 1e-9


def smooth_trajectory(
    start: Sequence[float], goal: Sequence[float], vel_scale: float, acc_scale: float = 1.0, n: int = 25
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """Cubic (smoothstep) joint-space interpolation with velocities; peak speed <= vel_scale * limit and
    peak acceleration <= acc_scale * limit (like MoveIt's time parameterisation)."""
    d = [b - a for a, b in zip(start, goal, strict=True)]
    vs = vel_scale if vel_scale > 0 else 1.0
    as_ = acc_scale if acc_scale > 0 else 1.0
    # smoothstep: peak ds/du = 1.5 -> 1.5 * |d_j| / T <= vs * vmax_j; peak d2s/du2 = 6 -> 6 |d_j| / T^2 <= as * amax_j
    duration = max(
        [1.5 * abs(dj) / (vs * vm) for dj, vm in zip(d, MAX_VEL, strict=True)]
        + [math.sqrt(6 * abs(dj) / (as_ * am)) for dj, am in zip(d, FR3_MAX_ACCELERATION, strict=True)]
        + [0.1]
    )
    pts, times, vels = [], [], []
    for i in range(n):
        u = i / (n - 1)
        s = 3 * u * u - 2 * u * u * u
        ds = (6 * u - 6 * u * u) / duration
        pts.append([a + s * dj for a, dj in zip(start, d, strict=True)])
        vels.append([ds * dj for dj in d])
        times.append(u * duration)
    pts[-1] = list(goal)
    return pts, times, vels


class FakeRosRobot:
    """See the module docstring. ``close()`` must be called (the fixture does it)."""

    def __init__(
        self,
        *,
        speedup: float = 1.0,
        publish_wrench: bool = True,
        reversed_joint_order: bool = False,
        prefix: str = "fr3",
        ee_frame: str | None = None,
        serve_moveit: bool = True,
    ) -> None:
        # prefix="panda" gives a Panda (same kinematics as the FR3) for tests against real MoveIt
        self.prefix = prefix
        self._timing = FakeBackend()  # only its trajectory timing helper is used
        self.joint_names = [n.replace("fr3", prefix, 1) for n in FR3_JOINT_NAMES]
        self.kin = FR3Kinematics(prefix=prefix)
        self.base_frame = f"{prefix}_link0"
        self.ee_frame = ee_frame or f"{prefix}_hand_tcp"
        self.arm_controller = f"{prefix}_arm_controller"
        self.finger_joints = [f"{prefix}_finger_joint1", f"{prefix}_finger_joint2"]
        self.q = list(FR3_READY)
        self.speedup = speedup
        self.lock = threading.Lock()
        self.publish_joint_states = True
        self.publish_wrench = publish_wrench
        self.force = [0.0, 0.0, 0.0]
        self.force_spike: tuple[float, float] | None = None  # (progress fraction, force_z N)
        self.reversed_joint_order = reversed_joint_order
        self.fail_next_goal = False
        self.gripper_width = 0.08
        self.object_width: float | None = None
        self.controllers = {
            "joint_state_broadcaster": ["joint_state_broadcaster/JointStateBroadcaster", "active"],
            self.arm_controller: ["joint_trajectory_controller/JointTrajectoryController", "active"],
            "cartesian_impedance_controller": [
                "franka_example_controllers/CartesianImpedanceController",
                "inactive",
            ],
        }
        # observations for assertions
        self.goals: list[Any] = []
        self.cancels = 0
        self.switch_requests: list[Any] = []
        self.plan_requests: list[Any] = []
        self.cartesian_requests: list[Any] = []
        self.fk_calls = 0
        self.collision_requests: list[Any] = []
        self.recoveries = 0
        self.gripper_goals: list[tuple[str, Any]] = []

        self.ctx = rclpy.Context()
        rclpy.init(context=self.ctx, signal_handler_options=SignalHandlerOptions.NO)
        self.node = node = Node(f"fake_{prefix}_cell", context=self.ctx)
        cb = ReentrantCallbackGroup()
        self.js_pub = node.create_publisher(JointStateMsg, "/joint_states", 10)
        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.wrench_pub = node.create_publisher(WrenchStamped, WRENCH_TOPIC, best_effort)
        self.gripper_js_pub = node.create_publisher(JointStateMsg, "/franka_gripper/joint_states", 1)
        self.image_pub = node.create_publisher(Image, "/camera/color/image_raw", 1)
        self.cimage_pub = node.create_publisher(CompressedImage, "/camera/color/image_raw/compressed", 1)
        self.tf = TransformBroadcaster(node)
        self.static_tf = StaticTransformBroadcaster(node)
        st = TransformStamped()
        st.header.stamp = node.get_clock().now().to_msg()
        st.header.frame_id, st.child_frame_id = "world", self.base_frame
        st.transform.rotation.w = 1.0
        self.static_tf.sendTransform(st)
        self._timers = [
            node.create_timer(0.01, self._publish_state, callback_group=cb),
            node.create_timer(0.1, self._publish_images, callback_group=cb),
        ]
        self._raw_image = self._make_raw_image(640, 480)

        node.create_service(
            ListControllers, "/controller_manager/list_controllers", self._list_controllers, callback_group=cb
        )
        node.create_service(
            SwitchController,
            "/controller_manager/switch_controller",
            self._switch_controllers,
            callback_group=cb,
        )
        if serve_moveit:
            node.create_service(GetPositionFK, "/compute_fk", self._compute_fk, callback_group=cb)
            node.create_service(GetMotionPlan, "/plan_kinematic_path", self._plan, callback_group=cb)
            node.create_service(
                GetCartesianPath, "/compute_cartesian_path", self._cartesian, callback_group=cb
            )
        self._servers = [
            ActionServer(
                node,
                FollowJointTrajectory,
                f"/{self.arm_controller}/follow_joint_trajectory",
                execute_callback=self._execute_trajectory,
                goal_callback=self._accept_trajectory,
                cancel_callback=lambda _gh: CancelResponse.ACCEPT,
                callback_group=cb,
            ),
            ActionServer(
                node,
                GripperCommand,
                "/franka_gripper/gripper_action",
                execute_callback=self._gripper_command,
                callback_group=cb,
            ),
        ]
        if HAVE_FRANKA_MSGS:
            for action_type, name, fn in (
                (Move, "/franka_gripper/move", self._gripper_move),
                (Grasp, "/franka_gripper/grasp", self._gripper_grasp),
                (Homing, "/franka_gripper/homing", self._gripper_homing),
                (ErrorRecovery, "/action_server/error_recovery", self._error_recovery),
            ):
                self._servers.append(
                    ActionServer(node, action_type, name, execute_callback=fn, callback_group=cb)
                )
            node.create_service(
                SetForceTorqueCollisionBehavior,
                "/service_server/set_force_torque_collision_behavior",
                self._set_collision,
                callback_group=cb,
            )

        self.executor = MultiThreadedExecutor(num_threads=8, context=self.ctx)
        self.executor.add_node(node)
        self.thread = threading.Thread(target=self._spin, daemon=True, name=f"fake-{prefix}-cell")
        self.thread.start()

    def _spin(self) -> None:
        with contextlib.suppress(Exception):  # ExternalShutdownException on close()
            self.executor.spin()

    def close(self) -> None:
        for timer in self._timers:  # stop periodic work first so no timer task outlives the node
            timer.cancel()
        time.sleep(0.02)
        self.executor.shutdown(timeout_sec=2.0)
        self.node.destroy_node()
        rclpy.try_shutdown(context=self.ctx)
        self.thread.join(timeout=2.0)

    # --- publishers -------------------------------------------------------------------
    def joints(self) -> list[float]:
        with self.lock:
            return list(self.q)

    def _publish_state(self) -> None:
        now = self.node.get_clock().now().to_msg()
        q = self.joints()
        if self.publish_joint_states:
            names, pos = list(self.joint_names), list(q)
            if self.reversed_joint_order:
                names, pos = names[::-1], pos[::-1]
            half = self.gripper_width / 2  # like franka's merged /joint_states: fingers included
            names, pos = [*names, *self.finger_joints], [*pos, half, half]
            js = JointStateMsg(name=names, position=pos, velocity=[0.0] * 9, effort=[0.0] * 9)
            js.header.stamp = now
            self.js_pub.publish(js)
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id, t.child_frame_id = self.base_frame, self.ee_frame
            m = self.kin.chain(q)[self.ee_frame]
            p, o = g.translation(m), g.quat_from_matrix(m)
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = o
            self.tf.sendTransform(t)
        if self.publish_wrench:
            w = WrenchStamped()
            w.header.stamp = now
            w.header.frame_id = f"{self.prefix}_EE"
            w.wrench.force.x, w.wrench.force.y, w.wrench.force.z = self.force
            self.wrench_pub.publish(w)
        half = self.gripper_width / 2
        self.gripper_js_pub.publish(JointStateMsg(name=self.finger_joints, position=[half, half]))

    @staticmethod
    def _make_raw_image(w: int, h: int) -> Image:
        row = bytearray()
        for x in range(w):
            row += bytes(((x * 255) // (w - 1), 0, 200))
        data = bytearray()
        for y in range(h):
            r = bytearray(row)
            r[1::3] = bytes([(y * 255) // (h - 1)]) * w
            data += r
        return Image(height=h, width=w, encoding="rgb8", is_bigendian=0, step=3 * w, data=bytes(data))

    def _publish_images(self) -> None:
        stamp = self.node.get_clock().now().to_msg()
        img = self._raw_image
        img.header.stamp = stamp
        img.header.frame_id = "camera_color_optical_frame"
        self.image_pub.publish(img)
        c = CompressedImage(format="png", data=gradient_png(64, 48))
        c.header.stamp = stamp
        self.cimage_pub.publish(c)

    # --- controller_manager ----------------------------------------------------------
    def _list_controllers(self, _req: Any, res: Any) -> Any:
        res.controller = [ControllerState(name=n, type=t, state=s) for n, (t, s) in self.controllers.items()]
        return res

    def _switch_controllers(self, req: Any, res: Any) -> Any:
        self.switch_requests.append(req)
        unknown = [
            n for n in [*req.activate_controllers, *req.deactivate_controllers] if n not in self.controllers
        ]
        if unknown:
            res.ok, res.message = False, f"unknown controllers {unknown}"
            return res
        for n in req.deactivate_controllers:
            self.controllers[n][1] = "inactive"
        for n in req.activate_controllers:
            self.controllers[n][1] = "active"
        res.ok, res.message = True, ""
        return res

    # --- MoveIt-like services ---------------------------------------------------------
    def _compute_fk(self, req: Any, res: Any) -> Any:
        self.fk_calls += 1
        names = list(req.robot_state.joint_state.name)
        q = [req.robot_state.joint_state.position[names.index(j)] for j in self.joint_names]
        chain = self.kin.chain(q)
        for link in req.fk_link_names:
            if link not in chain:
                res.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                return res
            m = chain[link]
            ps = PoseStamped()
            ps.header.frame_id = req.header.frame_id or self.base_frame
            (ps.pose.position.x, ps.pose.position.y, ps.pose.position.z) = g.translation(m)
            o = g.quat_from_matrix(m)
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = o
            res.pose_stamped.append(ps)
        res.fk_link_names = list(req.fk_link_names)
        res.error_code.val = MoveItErrorCodes.SUCCESS
        return res

    def _start_of(self, robot_state: Any) -> list[float]:
        names = list(robot_state.joint_state.name)
        if all(j in names for j in self.joint_names):
            return [robot_state.joint_state.position[names.index(j)] for j in self.joint_names]
        return self.joints()

    def _fill_trajectory(
        self, jt: Any, pts: list[list[float]], times: list[float], vels: list[list[float]] | None
    ) -> None:
        order = self.joint_names[::-1] if self.reversed_joint_order else self.joint_names
        idx = [self.joint_names.index(j) for j in order]
        jt.joint_names = list(order)
        out = []
        for k, (p, t) in enumerate(zip(pts, times, strict=True)):
            pt = JointTrajectoryPoint(positions=[p[i] for i in idx], time_from_start=_dur(t))
            if vels is not None:
                pt.velocities = [vels[k][i] for i in idx]
            out.append(pt)
        jt.points = out

    def _plan(self, req: Any, res: Any) -> Any:
        mpr = req.motion_plan_request
        self.plan_requests.append(mpr)
        out = res.motion_plan_response
        out.group_name = mpr.group_name
        if mpr.group_name != f"{self.prefix}_arm":
            out.error_code.val = MoveItErrorCodes.INVALID_GROUP_NAME
            return res
        start = self._start_of(mpr.start_state)
        goal_c = mpr.goal_constraints[0]
        if goal_c.joint_constraints:
            by_name = {c.joint_name: c.position for c in goal_c.joint_constraints}
            goal = [by_name[j] for j in self.joint_names]
        else:
            p = goal_c.position_constraints[0].constraint_region.primitive_poses[0].position
            o = goal_c.orientation_constraints[0].orientation
            target = g.matrix_from_quat((o.x, o.y, o.z, o.w), (p.x, p.y, p.z))
            ik = self.kin.ik(target, start)
            if not ik.success:
                out.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
                return res
            goal = ik.positions
        pts, times, vels = smooth_trajectory(
            start, goal, mpr.max_velocity_scaling_factor, mpr.max_acceleration_scaling_factor
        )
        self._fill_trajectory(out.trajectory.joint_trajectory, pts, times, vels)
        out.error_code.val = MoveItErrorCodes.SUCCESS
        return res

    def _cartesian(self, req: Any, res: Any) -> Any:
        self.cartesian_requests.append(req)
        q = self._start_of(req.start_state)
        cur = self.kin.fk(q)
        path = [list(q)]
        total = 0
        done = 0
        for wp in req.waypoints:
            tgt = g.matrix_from_quat(
                (wp.orientation.x, wp.orientation.y, wp.orientation.z, wp.orientation.w),
                (wp.position.x, wp.position.y, wp.position.z),
            )
            p0, p1 = g.translation(cur), g.translation(tgt)
            n = max(1, math.ceil(g.distance(p0, p1) / req.max_step))
            q0, q1 = g.quat_from_matrix(cur), g.quat_from_matrix(tgt)
            total += n
            for i in range(1, n + 1):
                s = i / n
                pos = [a + s * (b - a) for a, b in zip(p0, p1, strict=True)]
                ik = self.kin.ik(g.matrix_from_quat(g.slerp(q0, q1, s), pos), path[-1], restarts=0)
                if not ik.success:
                    break
                path.append(ik.positions)
                done += 1
            cur = tgt
        vs = getattr(req, "max_velocity_scaling_factor", 0.0) or 1.0
        as_ = getattr(req, "max_acceleration_scaling_factor", 0.0) or 1.0
        # rest-to-rest trapezoidal timing along the path, as MoveIt's time parameterisation does
        path, times = self._timing._time_parameterize_path(path, vs, as_)
        self._fill_trajectory(res.solution.joint_trajectory, path, times, None)
        res.fraction = done / total if total else 1.0
        res.error_code.val = MoveItErrorCodes.SUCCESS
        return res

    # --- joint_trajectory_controller ---------------------------------------------------
    def _accept_trajectory(self, _goal: Any) -> GoalResponse:
        if self.controllers[self.arm_controller][1] != "active":
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_trajectory(self, gh: Any) -> Any:
        traj = gh.request.trajectory
        self.goals.append(gh.request)
        result = FollowJointTrajectory.Result()
        names = list(traj.joint_names)
        if sorted(names) != sorted(self.joint_names):
            gh.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            return result
        idx = [names.index(j) for j in self.joint_names]
        pts = [(_s(p.time_from_start), [p.positions[i] for i in idx]) for p in traj.points]
        duration = pts[-1][0]
        if self.fail_next_goal:
            self.fail_next_goal = False
            time.sleep(0.05)
            gh.abort()
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = "fake: path tolerance violated"
            return result
        t0 = time.monotonic()
        while True:
            if gh.is_cancel_requested:
                self.cancels += 1
                gh.canceled()
                return result
            t = (time.monotonic() - t0) * self.speedup
            q = self._interp(pts, t)
            with self.lock:
                self.q = q
            if self.force_spike is not None and duration > 0 and t / duration >= self.force_spike[0]:
                self.force = [0.0, 0.0, self.force_spike[1]]
            fb = FollowJointTrajectory.Feedback()
            fb.joint_names = list(self.joint_names)
            fb.desired.positions = q
            fb.desired.time_from_start = _dur(min(t, duration))
            fb.actual.positions = q
            gh.publish_feedback(fb)
            if t >= duration:
                break
            time.sleep(0.005)
        gh.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    @staticmethod
    def _interp(pts: list[tuple[float, list[float]]], t: float) -> list[float]:
        if t <= pts[0][0]:
            return list(pts[0][1])
        for (ta, qa), (tb, qb) in itertools.pairwise(pts):
            if t <= tb:
                s = (t - ta) / (tb - ta) if tb > ta else 1.0
                return [a + s * (b - a) for a, b in zip(qa, qb, strict=True)]
        return list(pts[-1][1])

    # --- gripper / franka ---------------------------------------------------------------
    def _move_fingers(self, width: float) -> None:
        time.sleep(0.05)
        self.gripper_width = width

    def _gripper_move(self, gh: Any) -> Any:
        self.gripper_goals.append(("move", gh.request))
        self._move_fingers(gh.request.width)
        gh.succeed()
        return Move.Result(success=True, error="")

    def _gripper_grasp(self, gh: Any) -> Any:
        r = gh.request
        self.gripper_goals.append(("grasp", r))
        obj = self.object_width
        self._move_fingers(obj if obj is not None else 0.0)
        ok = obj is not None and r.width - r.epsilon.inner <= obj <= r.width + r.epsilon.outer
        if ok:
            gh.succeed()
        else:
            gh.abort()
        return Grasp.Result(success=ok, error="")

    def _gripper_homing(self, gh: Any) -> Any:
        self.gripper_goals.append(("homing", gh.request))
        self._move_fingers(0.08)
        gh.succeed()
        return Homing.Result(success=True, error="")

    def _gripper_command(self, gh: Any) -> Any:
        self.gripper_goals.append(("gripper_command", gh.request))
        width = 2 * gh.request.command.position
        obj = self.object_width
        stalled = gh.request.command.max_effort > 0 and obj is not None and width < obj
        self._move_fingers(obj if stalled else width)
        gh.succeed()
        return GripperCommand.Result(
            position=self.gripper_width, effort=0.0, stalled=stalled, reached_goal=not stalled
        )

    def _error_recovery(self, gh: Any) -> Any:
        self.recoveries += 1
        gh.succeed()
        return ErrorRecovery.Result()

    def _set_collision(self, req: Any, res: Any) -> Any:
        self.collision_requests.append(req)
        res.success, res.error = True, ""
        return res
