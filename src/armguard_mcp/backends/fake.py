"""Deterministic simulated Franka FR3 backend (no ROS required).

Used for tests, demos and CI. It has real FR3 kinematics, a simple planner, time-scaled
asynchronous execution (``speedup`` x faster than real time), a virtual table for contact
forces, a Franka Hand model, generated camera images and a fake ros2_control graph.
It does NOT model dynamics, self-collision or the Franka reflex behaviour beyond a
simple force threshold.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import anyio

from armguard_mcp import geometry as g
from armguard_mcp.backends.base import (
    AbortCheck,
    BackendFailed,
    CameraFrame,
    NotSupported,
    ProgressCallback,
    RobotBackend,
)
from armguard_mcp.imaging import gradient_png
from armguard_mcp.kinematics import FR3_JOINT_LIMITS, FR3_JOINT_NAMES, FR3_READY, FR3Kinematics
from armguard_mcp.models import (
    ControllerInfo,
    ExecutionResult,
    GraphInfo,
    GripperState,
    JointState,
    Plan,
    PlanKind,
    Pose,
    Quaternion,
    Vector3,
    Wrench,
)
from armguard_mcp.plans import new_plan_id

# Published FR3 joint velocity / acceleration limits (rad/s, rad/s^2).
FR3_MAX_VELOCITY = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)
FR3_MAX_ACCELERATION = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)

_ARM_CONTROLLERS = ("fr3_arm_controller", "cartesian_impedance_controller", "joint_impedance_controller")


@dataclass
class _Controller:
    name: str
    type: str
    state: str


class FakeBackend(RobotBackend):
    name = "fake"

    def __init__(
        self,
        *,
        joint_names: Sequence[str] = FR3_JOINT_NAMES,
        joint_limits: Sequence[tuple[float, float]] = FR3_JOINT_LIMITS,
        max_velocity: Sequence[float] = FR3_MAX_VELOCITY,
        max_acceleration: Sequence[float] = FR3_MAX_ACCELERATION,
        initial_positions: Sequence[float] = FR3_READY,
        speedup: float = 20.0,
        tick_s: float = 0.002,
        sample_dt: float = 0.05,
        table_height: float | None = None,
        table_stiffness_n_per_m: float = 2000.0,
        object_width: float | None = None,
        camera_topics: Sequence[str] = ("/camera/color/image_raw", "/wrist_camera/color/image_raw"),
        image_size: tuple[int, int] = (64, 48),
        trajectory_controller: str = "fr3_arm_controller",
        start_tolerance_rad: float = 0.01,
    ) -> None:
        if len(joint_names) != 7:
            raise ValueError("the fake FR3 backend has exactly 7 joints")
        self.joint_names = list(joint_names)
        self.kin = FR3Kinematics(
            prefix=joint_names[0].rsplit("_joint", 1)[0], joint_limits=tuple(joint_limits)
        )
        self.max_velocity = list(max_velocity)
        self.max_acceleration = list(max_acceleration)
        self.speedup = speedup
        self.tick_s = tick_s
        self.sample_dt = sample_dt
        self.table_height = table_height
        self.table_stiffness = table_stiffness_n_per_m
        self.object_width = object_width
        self.camera_topics = list(camera_topics)
        self.image_size = image_size
        self.trajectory_controller = trajectory_controller
        self.start_tolerance_rad = start_tolerance_rad
        self.gripper_max_width = 0.08

        self._q = [float(v) for v in initial_positions]
        self._dq = [0.0] * 7
        self._gripper_width = self.gripper_max_width
        self._grasped = False
        self._busy = False
        self._stop_requested = False
        self._gripper_stop_requested = False
        self._in_error = False
        self._collision_force_n = 100.0
        self._collision_torque_nm = 30.0
        self._started = False
        self._snapshots = 0
        self._controllers = {
            "joint_state_broadcaster": _Controller(
                "joint_state_broadcaster", "joint_state_broadcaster/JointStateBroadcaster", "active"
            ),
            "franka_robot_state_broadcaster": _Controller(
                "franka_robot_state_broadcaster",
                "franka_robot_state_broadcaster/FrankaRobotStateBroadcaster",
                "active",
            ),
            "fr3_arm_controller": _Controller(
                "fr3_arm_controller", "joint_trajectory_controller/JointTrajectoryController", "active"
            ),
            "cartesian_impedance_controller": _Controller(
                "cartesian_impedance_controller",
                "franka_example_controllers/CartesianImpedanceController",
                "inactive",
            ),
            "joint_impedance_controller": _Controller(
                "joint_impedance_controller",
                "franka_example_controllers/JointImpedanceExampleController",
                "inactive",
            ),
        }

    @classmethod
    def from_policy(cls, policy: object, **kwargs: object) -> FakeBackend:
        """Build a fake FR3 whose limits and home pose match ``policy`` (an armguard Policy)."""
        from armguard_mcp.policy import Policy

        assert isinstance(policy, Policy)
        r = policy.robot
        lims = r.limits_list()
        defaults: dict[str, object] = {
            "joint_names": r.joint_names,
            "joint_limits": [(lim.min, lim.max) for lim in lims],
            "max_velocity": [lim.max_velocity for lim in lims],
            "max_acceleration": [lim.max_acceleration for lim in lims],
            "initial_positions": r.home_joint_positions,
            "camera_topics": policy.perception.camera_topics or ("/camera/color/image_raw",),
            "start_tolerance_rad": policy.motion.start_tolerance_rad,
        }
        defaults.update(kwargs)
        return cls(**defaults)  # type: ignore[arg-type]

    # --- helpers -----------------------------------------------------------------------
    @property
    def base_frame(self) -> str:
        return self.kin.base_frame

    def _pose_from_matrix(self, t: g.Mat4, frame: str | None = None) -> Pose:
        return Pose(
            frame_id=frame or self.base_frame,
            position=Vector3.of(g.translation(t)),
            orientation=Quaternion.of(g.quat_from_matrix(t)),
        )

    def _tcp_z(self, q: Sequence[float]) -> float:
        return self.kin.fk(q)[2][3]

    def _contact_force(self, q: Sequence[float]) -> float:
        if self.table_height is None:
            return 0.0
        pen = self.table_height - self._tcp_z(q)
        return self.table_stiffness * pen if pen > 0 else 0.0

    def set_joint_positions(self, q: Sequence[float]) -> None:
        """Test hook: teleport the simulated robot (e.g. to make a plan stale)."""
        self._q = [float(v) for v in q]

    # --- lifecycle ---------------------------------------------------------------------
    async def start(self) -> None:
        self._started = True

    async def shutdown(self) -> None:
        self._stop_requested = True
        self._started = False

    # --- state -------------------------------------------------------------------------
    async def get_joint_state(self) -> JointState:
        return JointState(
            names=list(self.joint_names),
            positions=list(self._q),
            velocities=list(self._dq),
            efforts=[0.0] * 7,
            stamp=time.time(),
        )

    async def get_ee_pose(self) -> Pose:
        return self._pose_from_matrix(self.kin.fk(self._q))

    async def forward_kinematics(self, joint_positions: Sequence[float]) -> Pose:
        if len(joint_positions) != 7:
            raise BackendFailed(f"forward_kinematics needs 7 joint positions, got {len(joint_positions)}")
        return self._pose_from_matrix(self.kin.fk(joint_positions))

    def _frames(self) -> dict[str, g.Mat4]:
        frames = self.kin.chain(self._q)
        frames["world"] = g.identity()
        return frames

    async def lookup_transform(self, target_frame: str, source_frame: str) -> Pose:
        frames = self._frames()
        missing = [f for f in (target_frame, source_frame) if f not in frames]
        if missing:
            raise BackendFailed(f"unknown frame(s) {missing}; known frames: {sorted(frames)}")
        t = g.matmul(g.invert(frames[target_frame]), frames[source_frame])
        return self._pose_from_matrix(t, frame=target_frame)

    async def get_wrench(self) -> Wrench | None:
        f = self._contact_force(self._q)
        return Wrench(
            frame_id=self.base_frame,
            force=Vector3(x=0.0, y=0.0, z=f),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
            stamp=time.time(),
        )

    # --- controllers -------------------------------------------------------------------
    async def list_controllers(self) -> list[ControllerInfo]:
        return [ControllerInfo(name=c.name, type=c.type, state=c.state) for c in self._controllers.values()]

    async def switch_controllers(
        self, activate: Sequence[str], deactivate: Sequence[str]
    ) -> list[ControllerInfo]:
        unknown = [n for n in [*activate, *deactivate] if n not in self._controllers]
        if unknown:
            raise BackendFailed(f"unknown controller(s): {unknown}")
        if self._busy:
            raise BackendFailed("cannot switch controllers while a trajectory is executing")
        new_state = {n: c.state for n, c in self._controllers.items()}
        for n in deactivate:
            new_state[n] = "inactive"
        for n in activate:
            new_state[n] = "active"
        active_arm = [n for n in _ARM_CONTROLLERS if new_state.get(n) == "active"]
        if len(active_arm) > 1:
            raise BackendFailed(
                f"controllers {active_arm} would claim the same arm joints; deactivate one in the same call"
            )
        for n, s in new_state.items():
            self._controllers[n].state = s
        return await self.list_controllers()

    # --- planning ----------------------------------------------------------------------
    @staticmethod
    def _trapezoid(length: float, v_max: float, a_max: float) -> tuple[float, Callable[[float], float]]:
        """Rest-to-rest trapezoidal (or triangular) profile over ``length``: (duration, t_of_s)."""
        if v_max * v_max / a_max >= length:  # triangle profile
            t_acc = math.sqrt(length / a_max)
            v_peak, t_cruise = a_max * t_acc, 0.0
        else:
            t_acc = v_max / a_max
            v_peak, t_cruise = v_max, length / v_max - v_max / a_max
        duration = 2 * t_acc + t_cruise
        s_acc = 0.5 * a_max * t_acc * t_acc

        def t_of_s(s: float) -> float:
            if s <= s_acc:
                return math.sqrt(2 * max(s, 0.0) / a_max)
            if s <= length - s_acc:
                return t_acc + (s - s_acc) / v_peak
            return duration - math.sqrt(2 * max(length - s, 0.0) / a_max)

        return duration, t_of_s

    def _time_parameterize(
        self, start: Sequence[float], goal: Sequence[float], vel_scale: float, acc_scale: float
    ) -> tuple[list[list[float]], list[float]]:
        """Synchronised trapezoidal profile along the straight joint-space line start->goal."""
        d = [b - a for a, b in zip(start, goal, strict=True)]
        moving = [(abs(dj), j) for j, dj in enumerate(d) if abs(dj) > 1e-9]
        if not moving:
            return [list(start), list(goal)], [0.0, self.sample_dt]
        v_s = min(self.max_velocity[j] * vel_scale / dj for dj, j in moving)  # max ds/dt
        a_s = min(self.max_acceleration[j] * acc_scale / dj for dj, j in moving)  # max d2s/dt2
        if v_s * v_s / a_s >= 1.0:  # triangle profile
            t_acc = math.sqrt(1.0 / a_s)
            v_peak, duration, t_cruise = a_s * t_acc, 2 * t_acc, 0.0
        else:
            t_acc = v_s / a_s
            v_peak = v_s
            t_cruise = 1.0 / v_s - t_acc
            duration = 2 * t_acc + t_cruise

        def s_of(t: float) -> float:
            if t <= t_acc:
                return 0.5 * a_s * t * t
            if t <= t_acc + t_cruise:
                return 0.5 * a_s * t_acc * t_acc + v_peak * (t - t_acc)
            td = duration - t
            return 1.0 - 0.5 * a_s * td * td

        n = max(2, math.ceil(duration / self.sample_dt) + 1)
        times = [duration * i / (n - 1) for i in range(n)]
        pts = [[a + s_of(t) * dj for a, dj in zip(start, d, strict=True)] for t in times]
        pts[-1] = list(goal)
        return pts, times

    def _time_parameterize_path(
        self, path: list[list[float]], vel_scale: float, acc_scale: float
    ) -> tuple[list[list[float]], list[float]]:
        """Rest-to-rest trapezoidal timing along a joint-space polyline (the Cartesian planner's output).

        The path parameter s is "seconds at full joint speed": segment k has length
        max_j |dq_j| / v_max_j. Then ds/dt <= vel_scale keeps every joint under vel_scale * v_max, and
        d2s/dt2 is bounded so every joint stays under acc_scale * a_max along each segment. The path
        is curved in joint space, so the direction changes between segments add acceleration; if the
        result is still above acc_scale, the whole profile is slowed down uniformly (stretching time
        by k divides every acceleration by k^2).
        """
        pts = [path[0]]
        for q in path[1:]:
            if max(abs(a - b) for a, b in zip(q, pts[-1], strict=True)) > 1e-9:
                pts.append(q)
        if len(pts) < 2:
            return [list(path[0]), list(path[-1])], [0.0, self.sample_dt]
        seg: list[float] = []
        a_max = math.inf
        for a, b in itertools.pairwise(pts):
            dq = [abs(y - x) for x, y in zip(a, b, strict=True)]
            ell = max(d / self.max_velocity[j] for j, d in enumerate(dq))
            seg.append(ell)
            for j, d in enumerate(dq):
                if d > 1e-12:
                    a_max = min(a_max, self.max_acceleration[j] * acc_scale * ell / d)
        duration, t_of_s = self._trapezoid(sum(seg), vel_scale, a_max)
        times, s = [0.0], 0.0
        for ell in seg:
            s += ell
            times.append(max(t_of_s(s), times[-1] + 1e-4))
        times[-1] = max(times[-1], duration)
        ratio = self._accel_ratio(pts, times)
        if ratio > acc_scale:
            k = math.sqrt(ratio / acc_scale) * 1.01
            times = [t * k for t in times]
        return [list(q) for q in pts], times

    def _accel_ratio(self, pts: list[list[float]], times: list[float]) -> float:
        """Peak joint acceleration / limit from segment-average velocities (rest at both ends)."""
        vel = [[0.0] * 7]
        mids = [times[0]]
        for i in range(1, len(pts)):
            dt = times[i] - times[i - 1]
            vel.append([(pts[i][j] - pts[i - 1][j]) / dt for j in range(7)])
            mids.append((times[i] + times[i - 1]) / 2)
        vel.append([0.0] * 7)
        mids.append(times[-1])
        return max(
            abs(vel[k][j] - vel[k - 1][j]) / (mids[k] - mids[k - 1]) / self.max_acceleration[j]
            for k in range(1, len(vel))
            for j in range(7)
        )

    def _make_plan(
        self,
        kind: PlanKind,
        waypoints: list[list[float]],
        times: list[float],
        vel_scale: float,
        acc_scale: float,
        source: str,
    ) -> Plan:
        return Plan(
            plan_id=new_plan_id(),
            kind=kind,
            joint_names=list(self.joint_names),
            waypoints=waypoints,
            time_from_start=times,
            duration_s=times[-1],
            created_at=time.time(),
            source_request=source,
            velocity_scaling=vel_scale,
            acceleration_scaling=acc_scale,
        )

    async def plan_to_joints(self, target: Sequence[float], vel_scale: float, acc_scale: float) -> Plan:
        if len(target) != 7:
            raise BackendFailed(f"expected 7 joint positions, got {len(target)}")
        pts, times = self._time_parameterize(self._q, list(target), vel_scale, acc_scale)
        return self._make_plan(
            "joints", pts, times, vel_scale, acc_scale, f"joints -> {[round(v, 4) for v in target]}"
        )

    def _pose_matrix(self, pose: Pose) -> g.Mat4:
        if pose.frame_id and pose.frame_id != self.base_frame:
            raise BackendFailed(f"pose must be in {self.base_frame}, got {pose.frame_id!r}")
        return g.matrix_from_quat(pose.orientation.as_tuple(), pose.position.as_tuple())

    async def plan_to_pose(self, pose: Pose, vel_scale: float, acc_scale: float) -> Plan:
        target = self._pose_matrix(pose)
        res = await anyio.to_thread.run_sync(lambda: self.kin.ik(target, self._q))
        if not res.success:
            raise BackendFailed(
                f"no IK solution within joint limits (best position error {res.position_error_m * 1000:.1f} mm)"
            )
        pts, times = self._time_parameterize(self._q, res.positions, vel_scale, acc_scale)
        p = pose.position
        return self._make_plan(
            "pose", pts, times, vel_scale, acc_scale, f"pose -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )

    async def plan_cartesian(
        self, waypoints: Sequence[Pose], max_step: float, vel_scale: float, acc_scale: float
    ) -> Plan:
        if not waypoints:
            raise BackendFailed("plan_cartesian needs at least one waypoint")
        if max_step <= 0:
            raise BackendFailed("max_step must be > 0")
        targets = [self._pose_matrix(p) for p in waypoints]

        def solve() -> list[list[float]]:
            q = list(self._q)
            cur = self.kin.fk(q)
            path = [list(q)]
            for tgt in targets:
                p0, p1 = g.translation(cur), g.translation(tgt)
                q0, q1 = g.quat_from_matrix(cur), g.quat_from_matrix(tgt)
                n = max(
                    1, math.ceil(max(g.distance(p0, p1) / max_step, g.angle_between_quats(q0, q1) / 0.05))
                )
                for i in range(1, n + 1):
                    s = i / n
                    pos = [a + s * (b - a) for a, b in zip(p0, p1, strict=True)]
                    t = g.matrix_from_quat(g.slerp(q0, q1, s), pos)
                    res = self.kin.ik(t, q, restarts=0)
                    if not res.success:
                        raise BackendFailed(
                            f"Cartesian path not feasible: IK failed at {100 * ((len(path) - 1) / max(1, n)):.0f}% "
                            f"of segment (position error {res.position_error_m * 1000:.1f} mm)"
                        )
                    if max(abs(a - b) for a, b in zip(res.positions, q, strict=True)) > 0.3:
                        raise BackendFailed(
                            "Cartesian path not feasible: joint-space jump (singularity or flip)"
                        )
                    q = res.positions
                    path.append(list(q))
                cur = tgt
            return path

        path = await anyio.to_thread.run_sync(solve)
        path, times = self._time_parameterize_path(path, vel_scale, acc_scale)
        return self._make_plan(
            "cartesian", path, times, vel_scale, acc_scale, f"cartesian, {len(waypoints)} waypoint(s)"
        )

    # --- execution ---------------------------------------------------------------------
    @staticmethod
    def _interp(plan: Plan, t: float) -> list[float]:
        ts = plan.time_from_start
        if t <= ts[0]:
            return list(plan.waypoints[0])
        for i in range(1, len(ts)):
            if t <= ts[i]:
                s = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
                a, b = plan.waypoints[i - 1], plan.waypoints[i]
                return [x + s * (y - x) for x, y in zip(a, b, strict=True)]
        return list(plan.waypoints[-1])

    async def execute(
        self, plan: Plan, on_progress: ProgressCallback, should_abort: AbortCheck
    ) -> ExecutionResult:
        if self._busy:
            raise BackendFailed("another trajectory is already executing")
        if self._in_error:
            raise BackendFailed("robot is in reflex/error mode; call error_recovery first")
        if self._controllers[self.trajectory_controller].state != "active":
            raise BackendFailed(
                f"{self.trajectory_controller} is not active; cannot execute a joint trajectory"
            )
        if plan.joint_names != self.joint_names:
            raise BackendFailed("plan joint names do not match the robot")
        dev = max(abs(a - b) for a, b in zip(plan.start, self._q, strict=True))
        if dev > self.start_tolerance_rad:
            raise BackendFailed(f"robot is {dev:.4f} rad away from the plan's start state")
        # Like a JointTrajectoryController, start from the actual state (within the tolerance).
        plan = plan.model_copy(update={"waypoints": [list(self._q), *plan.waypoints[1:]]})

        self._busy = True
        self._stop_requested = False
        max_force = 0.0
        next_report = 0.0
        t0 = time.monotonic()
        prev_q, prev_t = list(self._q), 0.0
        try:
            while True:
                sim_t = (time.monotonic() - t0) * self.speedup
                done = sim_t >= plan.duration_s
                q = list(plan.final) if done else self._interp(plan, sim_t)
                if sim_t > prev_t:
                    self._dq = [(b - a) / (sim_t - prev_t) for a, b in zip(prev_q, q, strict=True)]
                prev_q, prev_t = q, sim_t
                self._q = q
                force = self._contact_force(q)
                max_force = max(max_force, force)
                if force > self._collision_force_n:
                    self._in_error = True
                    return self._finish(
                        False, f"collision reflex: {force:.1f} N > robot threshold", max_force, True
                    )
                if done:
                    await on_progress(1.0, "trajectory complete")
                    return self._finish(True, "trajectory complete", max_force, False)
                if self._stop_requested or should_abort():
                    return self._finish(
                        False, f"aborted at t={sim_t:.2f}/{plan.duration_s:.2f} s", max_force, True
                    )
                frac = sim_t / plan.duration_s if plan.duration_s > 0 else 1.0
                if frac >= next_report:
                    await on_progress(frac, f"executing {plan.plan_id}: {100 * frac:.0f}%")
                    next_report = frac + 0.1
                await anyio.sleep(self.tick_s)
        finally:
            self._busy = False
            self._dq = [0.0] * 7

    def _finish(self, success: bool, message: str, max_force: float, aborted: bool) -> ExecutionResult:
        self._dq = [0.0] * 7
        return ExecutionResult(
            success=success,
            message=message,
            final_joint_state=JointState(
                names=list(self.joint_names), positions=list(self._q), stamp=time.time()
            ),
            max_observed_force_n=max_force,
            aborted=aborted,
        )

    async def stop(self) -> None:
        """Stop the arm trajectory and the gripper."""
        self._stop_requested = True
        self._gripper_stop_requested = True

    # --- gripper -----------------------------------------------------------------------
    def _gripper_state(self) -> GripperState:
        return GripperState(
            width_m=self._gripper_width,
            max_width_m=self.gripper_max_width,
            is_grasped=self._grasped,
            stamp=time.time(),
        )

    async def _gripper_travel(self, target: float, speed: float, what: str) -> None:
        """Move the fingers towards ``target`` in ticks; ``stop()`` halts them where they are."""
        self._gripper_stop_requested = False
        start = self._gripper_width
        duration = abs(target - start) / max(speed, 1e-3) / self.speedup
        t0 = time.monotonic()
        while True:
            frac = min(1.0, (time.monotonic() - t0) / duration) if duration > 0 else 1.0
            self._gripper_width = start + frac * (target - start)
            if frac >= 1.0:
                return
            if self._gripper_stop_requested:
                self._grasped = False
                raise BackendFailed(f"{what} stopped at {self._gripper_width:.4f} m (stop requested)")
            await anyio.sleep(self.tick_s)

    async def gripper_move(self, width: float, speed: float) -> GripperState:
        if not 0.0 <= width <= self.gripper_max_width:
            raise BackendFailed(f"width {width} outside [0, {self.gripper_max_width}] m")
        if self.object_width is not None and self._grasped and width < self.object_width:
            raise BackendFailed("move blocked by grasped object; open the gripper first")
        await self._gripper_travel(width, speed, "gripper move")
        self._gripper_width = width
        self._grasped = False
        return self._gripper_state()

    async def gripper_grasp(
        self, width: float, force: float, speed: float, epsilon_inner: float, epsilon_outer: float
    ) -> GripperState:
        final = self.object_width if self.object_width is not None else 0.0
        await self._gripper_travel(final, speed, "gripper grasp")
        self._gripper_width = final
        self._grasped = (
            self.object_width is not None
            and width - epsilon_inner <= self.object_width <= width + epsilon_outer
        )
        return self._gripper_state()

    async def gripper_home(self) -> GripperState:
        return await self.gripper_move(self.gripper_max_width, 0.1)

    async def get_gripper_state(self) -> GripperState | None:
        return self._gripper_state()

    # --- perception --------------------------------------------------------------------
    async def camera_snapshot(self, topic: str, max_width: int) -> CameraFrame:
        if topic not in self.camera_topics:
            raise BackendFailed(f"no image received on {topic} (known: {self.camera_topics})")
        self._snapshots += 1
        w, h = self.image_size
        return CameraFrame("image/png", gradient_png(w, h, seed=self._snapshots), w, h, time.time())

    # --- robot-specific ----------------------------------------------------------------
    async def set_collision_thresholds(self, force_n: float, torque_nm: float) -> None:
        if self._busy:
            raise BackendFailed("cannot change collision thresholds while moving")
        self._collision_force_n = force_n
        self._collision_torque_nm = torque_nm

    async def error_recovery(self) -> None:
        if self._busy:
            raise NotSupported("error recovery is not possible while a trajectory is executing")
        self._in_error = False

    async def list_graph(self) -> GraphInfo:
        return GraphInfo(
            nodes=[
                "/controller_manager",
                "/franka_gripper",
                "/move_group",
                "/robot_state_publisher",
                "/fr3_arm_controller",
                "/franka_robot_state_broadcaster",
                "/joint_state_broadcaster",
            ],
            topics=[
                "/joint_states",
                "/franka_robot_state_broadcaster/robot_state",
                "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
                "/franka_gripper/joint_states",
                "/tf",
                "/tf_static",
                *self.camera_topics,
            ],
            services=[
                "/controller_manager/list_controllers",
                "/controller_manager/switch_controller",
                "/service_server/set_full_collision_behavior",
                "/compute_ik",
                "/compute_fk",
            ],
            actions=[
                "/fr3_arm_controller/follow_joint_trajectory",
                "/franka_gripper/move",
                "/franka_gripper/grasp",
                "/franka_gripper/homing",
                "/move_action",
                "/action_server/error_recovery",
            ],
        )
