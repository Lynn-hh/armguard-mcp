"""Ros2Backend against real rclpy endpoints (the in-process fake FR3 cell)."""

from __future__ import annotations

import math
import time

import anyio
import pytest

pytest.importorskip("rclpy", reason="needs a sourced ROS 2 environment (rclpy)")


from armguard_mcp import geometry as g
from armguard_mcp.backends.base import BackendFailed, BackendTimeout, NotSupported
from armguard_mcp.kinematics import FR3_READY, FR3Kinematics
from armguard_mcp.models import Pose, Quaternion, Vector3
from tests_ros.conftest import ros_policy, running_backend
from tests_ros.fake_robot import HAVE_FRANKA_MSGS, FakeRosRobot

pytestmark = pytest.mark.anyio

READY = list(FR3_READY)
KIN = FR3Kinematics()


def target(j1: float) -> list[float]:
    q = list(READY)
    q[0] = j1
    return q


async def noop_progress(fraction: float, message: str) -> None:
    pass


async def test_startup_state_and_fk(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        assert b.availability and all(b.availability.values()), b.availability
        js = await b.get_joint_state()
        assert js.names == [f"fr3_joint{i}" for i in range(1, 8)]
        assert js.positions == pytest.approx(READY, abs=1e-9)
        ee = await b.get_ee_pose()
        assert ee.frame_id == "fr3_link0"
        assert (ee.position.x, ee.position.y, ee.position.z) == pytest.approx(
            (0.30689, 0.0, 0.48688), abs=1e-4
        )
        w = await b.get_wrench()
        assert w is not None and w.force_norm == pytest.approx(0.0) and w.frame_id == "fr3_EE"

        # FK through /compute_fk matches the local model and is cached
        q = target(0.4)
        pose = await b.forward_kinematics(q)
        t = KIN.fk(q)
        assert pose.position.as_tuple() == pytest.approx(tuple(g.translation(t)), abs=1e-9)
        n = robot.fk_calls
        await b.forward_kinematics(q)
        assert robot.fk_calls == n

        # tf lookups, including an unknown frame
        tf = await b.lookup_transform("world", "fr3_hand_tcp")
        assert tf.frame_id == "world" and tf.position.x == pytest.approx(0.30689, abs=1e-4)
        with pytest.raises(BackendFailed, match="no transform"):
            await b.lookup_transform("world", "no_such_frame")


async def test_fr3_analytic_fk_needs_no_service(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy({"fk_source": "fr3_analytic"})) as b:
        await b.forward_kinematics(target(0.2))
        assert robot.fk_calls == 0


async def test_joint_order_is_normalised() -> None:
    robot = FakeRosRobot(reversed_joint_order=True)
    try:
        async with running_backend(ros_policy()) as b:
            js = await b.get_joint_state()
            assert js.positions == pytest.approx(READY, abs=1e-9)
            plan = await b.plan_to_joints(target(0.1), 0.2, 0.2)
            assert plan.joint_names == js.names
            assert plan.final == pytest.approx(target(0.1), abs=1e-9)
            res = await b.execute(plan, noop_progress, lambda: False)
            assert res.success, res.message
            assert robot.joints() == pytest.approx(target(0.1), abs=1e-6)
    finally:
        robot.close()


async def test_stale_joint_states_are_refused(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy({"joint_state_max_age_s": 0.3})) as b:
        await b.get_joint_state()
        robot.publish_joint_states = False
        await anyio.sleep(0.4)
        with pytest.raises(BackendTimeout, match="stale"):
            await b.get_joint_state()


async def test_missing_wrench() -> None:
    robot = FakeRosRobot(publish_wrench=False)
    try:
        async with running_backend(ros_policy({"startup_timeout_s": 2.0})) as b:
            with pytest.raises(BackendTimeout, match="no fresh external wrench"):
                await b.get_wrench()
        async with running_backend(ros_policy({"require_wrench": False, "startup_timeout_s": 2.0})) as b:
            assert await b.get_wrench() is None
    finally:
        robot.close()


async def test_plan_and_execute_joint_goal(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        plan = await b.plan_to_joints(target(0.3), 0.2, 0.15)
        req = robot.plan_requests[-1]
        assert req.group_name == "fr3_arm"
        assert req.max_velocity_scaling_factor == pytest.approx(0.2)
        assert req.max_acceleration_scaling_factor == pytest.approx(0.15)
        assert list(req.start_state.joint_state.position) == pytest.approx(READY, abs=1e-9)
        jc = req.goal_constraints[0].joint_constraints
        assert [c.joint_name for c in jc] == plan.joint_names and jc[0].position == pytest.approx(0.3)
        assert plan.kind == "joints" and plan.start == pytest.approx(READY, abs=1e-9)
        assert plan.duration_s > 0.1

        progress: list[float] = []

        async def on_progress(fraction: float, message: str) -> None:
            progress.append(fraction)

        res = await b.execute(plan, on_progress, lambda: False)
        assert res.success and not res.aborted, res.message
        assert res.final_joint_state is not None
        assert res.final_joint_state.positions == pytest.approx(target(0.3), abs=1e-6)
        assert progress[-1] == 1.0 and len(progress) >= 3

        # the controller received exactly the validated points, with MoveIt's velocities attached
        goal = robot.goals[-1]
        assert list(goal.trajectory.joint_names) == plan.joint_names
        assert len(goal.trajectory.points) == len(plan.waypoints)
        for pt, q, t in zip(goal.trajectory.points, plan.waypoints, plan.time_from_start, strict=True):
            assert list(pt.positions) == pytest.approx(q, abs=1e-12)
            assert pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9 == pytest.approx(t, abs=1e-9)
            assert len(pt.velocities) == 7


async def test_execute_abort_cancels_goal(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        plan = await b.plan_to_joints(target(0.8), 0.1, 0.1)
        assert plan.duration_s > 2.0
        t0 = time.monotonic()
        res = await b.execute(plan, noop_progress, lambda: time.monotonic() - t0 > 0.5)
        assert res.aborted and not res.success
        assert robot.cancels == 1
        q = robot.joints()[0]
        assert 0.0 < q < 0.8
        await anyio.sleep(0.3)
        assert robot.joints()[0] == pytest.approx(q, abs=1e-9)  # the arm stays where it stopped


async def test_stop_cancels_goal_from_another_task(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        plan = await b.plan_to_joints(target(0.8), 0.1, 0.1)
        results = []

        async def run() -> None:
            results.append(await b.execute(plan, noop_progress, lambda: False))

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await anyio.sleep(0.5)
            await b.stop()
        assert results[0].aborted and robot.cancels >= 1
        await b.stop()  # nothing to stop: must not raise


async def test_cancelled_execute_task_stops_the_robot(robot: FakeRosRobot) -> None:
    """If the MCP call is cancelled (client gone, task group torn down), the goal is cancelled too."""
    async with running_backend(ros_policy()) as b:
        plan = await b.plan_to_joints(target(0.8), 0.1, 0.1)
        with anyio.move_on_after(0.5) as scope:
            await b.execute(plan, noop_progress, lambda: False)
        assert scope.cancelled_caught
        await anyio.sleep(0.2)
        assert robot.cancels == 1
        q = robot.joints()[0]
        await anyio.sleep(0.3)
        assert robot.joints()[0] == pytest.approx(q, abs=1e-9) and q < 0.8


async def test_controller_failure_is_reported(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        plan = await b.plan_to_joints(target(0.1), 0.2, 0.2)
        robot.fail_next_goal = True
        res = await b.execute(plan, noop_progress, lambda: False)
        assert not res.success and "ABORTED" in res.message and "path tolerance" in res.message


async def test_inactive_controller_rejects_goal(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        plan = await b.plan_to_joints(target(0.1), 0.2, 0.2)
        await b.switch_controllers(["cartesian_impedance_controller"], ["fr3_arm_controller"])
        with pytest.raises(BackendFailed, match="rejected"):
            await b.execute(plan, noop_progress, lambda: False)


async def test_plan_to_pose_and_cartesian(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        ee = await b.get_ee_pose()
        goal = Pose(
            frame_id="fr3_link0",
            position=Vector3(x=ee.position.x + 0.05, y=ee.position.y, z=ee.position.z - 0.05),
            orientation=ee.orientation,
        )
        plan = await b.plan_to_pose(goal, 0.1, 0.1)
        req = robot.plan_requests[-1]
        pc = req.goal_constraints[0].position_constraints[0]
        assert pc.link_name == "fr3_hand_tcp" and pc.header.frame_id == "fr3_link0"
        assert pc.constraint_region.primitive_poses[0].position.x == pytest.approx(goal.position.x)
        fk = await b.forward_kinematics(plan.final)
        assert fk.position.as_tuple() == pytest.approx(goal.position.as_tuple(), abs=1e-3)

        down = Pose(
            frame_id="fr3_link0",
            position=Vector3(x=ee.position.x, y=ee.position.y, z=ee.position.z - 0.05),
            orientation=ee.orientation,
        )
        cplan = await b.plan_cartesian([down], 0.005, 0.1, 0.1)
        creq = robot.cartesian_requests[-1]
        assert creq.max_step == pytest.approx(0.005) and creq.link_name == "fr3_hand_tcp"
        assert creq.revolute_jump_threshold == pytest.approx(0.3)
        assert creq.max_velocity_scaling_factor == pytest.approx(0.1) and creq.avoid_collisions
        assert cplan.kind == "cartesian" and len(cplan.waypoints) >= 10
        res = await b.execute(cplan, noop_progress, lambda: False)
        assert res.success, res.message
        final = await b.get_ee_pose()
        assert final.position.z == pytest.approx(ee.position.z - 0.05, abs=2e-3)

        with pytest.raises(BackendFailed, match="must be in fr3_link0"):
            await b.plan_to_pose(
                Pose(frame_id="world", position=goal.position, orientation=Quaternion()), 0.1, 0.1
            )


async def test_partial_cartesian_path_is_refused(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        ee = await b.get_ee_pose()
        unreachable = Pose(
            frame_id="fr3_link0",
            position=Vector3(x=1.5, y=0.0, z=ee.position.z),
            orientation=ee.orientation,
        )
        with pytest.raises(BackendFailed, match="Cartesian path not feasible"):
            await b.plan_cartesian([unreachable], 0.01, 0.1, 0.1)


async def test_controllers(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        names = {c.name: c.state for c in await b.list_controllers()}
        assert names["fr3_arm_controller"] == "active"
        after = {
            c.name: c.state
            for c in await b.switch_controllers(["cartesian_impedance_controller"], ["fr3_arm_controller"])
        }
        assert (
            after["cartesian_impedance_controller"] == "active" and after["fr3_arm_controller"] == "inactive"
        )
        req = robot.switch_requests[-1]
        assert req.strictness == req.STRICT and req.activate_asap and req.timeout.sec == 5
        with pytest.raises(BackendFailed, match="unknown controllers"):
            await b.switch_controllers(["nope"], [])


async def test_camera_raw_and_compressed(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        frame = await b.camera_snapshot("/camera/color/image_raw", 320)
        assert frame.mime == "image/png" and frame.data[:4] == b"\x89PNG"
        assert frame.width == 320 and frame.height == 240
        c = await b.camera_snapshot("/camera/color/image_raw/compressed", 320)
        assert c.mime == "image/png" and (c.width, c.height) == (64, 48)
        with pytest.raises(BackendFailed, match="no publisher"):
            await b.camera_snapshot("/nope/image", 320)


@pytest.mark.skipif(not HAVE_FRANKA_MSGS, reason="franka_msgs not built")
async def test_franka_gripper(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy({"gripper_interface": "franka"})) as b:
        st = await b.gripper_move(0.04, 0.05)
        assert st.width_m == pytest.approx(0.04) and not st.is_grasped
        assert robot.gripper_goals[-1][0] == "move" and robot.gripper_goals[-1][1].speed == pytest.approx(
            0.05
        )
        robot.object_width = 0.03
        st = await b.gripper_grasp(0.03, 20.0, 0.05, 0.005, 0.005)
        assert st.is_grasped and st.width_m == pytest.approx(0.03)
        grasp = robot.gripper_goals[-1][1]
        assert grasp.force == pytest.approx(20.0) and grasp.epsilon.outer == pytest.approx(0.005)
        robot.object_width = None
        st = await b.gripper_grasp(0.03, 20.0, 0.05, 0.005, 0.005)
        assert not st.is_grasped  # nothing between the fingers: not an error, just no grasp
        st = await b.gripper_home()
        assert st.width_m == pytest.approx(0.08)
        assert (await b.get_gripper_state()).width_m == pytest.approx(0.08)


async def test_gripper_command_fallback(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy({"gripper_interface": "gripper_command"})) as b:
        st = await b.gripper_move(0.05, 0.05)
        kind, goal = robot.gripper_goals[-1]
        assert kind == "gripper_command" and goal.command.position == pytest.approx(0.025)
        assert st.width_m == pytest.approx(0.05)
        robot.object_width = 0.03
        st = await b.gripper_grasp(0.02, 15.0, 0.05, 0.005, 0.005)
        assert st.is_grasped and robot.gripper_goals[-1][1].command.max_effort == pytest.approx(15.0)


@pytest.mark.skipif(not HAVE_FRANKA_MSGS, reason="franka_msgs not built")
async def test_franka_collision_thresholds_and_recovery(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        await b.set_collision_thresholds(30.0, 8.0)
        req = robot.collision_requests[-1]
        assert list(req.upper_force_thresholds_nominal) == [30.0, 30.0, 30.0, 8.0, 8.0, 8.0]
        assert list(req.lower_torque_thresholds_nominal) == [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        await b.error_recovery()
        assert robot.recoveries == 1


async def test_franka_services_can_be_disabled(robot: FakeRosRobot) -> None:
    cfg = {"collision_behavior_service": None, "error_recovery_action": None, "gripper_interface": "none"}
    async with running_backend(ros_policy(cfg)) as b:
        with pytest.raises(NotSupported):
            await b.set_collision_thresholds(30.0, 8.0)
        with pytest.raises(NotSupported):
            await b.error_recovery()
        with pytest.raises(NotSupported):
            await b.gripper_move(0.04, 0.05)
        assert await b.get_gripper_state() is None


async def test_graph(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        graph = await b.list_graph()
        assert "/fake_fr3_cell" in graph.nodes and "/armguard_mcp" not in graph.nodes
        assert "/joint_states" in graph.topics
        assert "/fr3_arm_controller/follow_joint_trajectory" in graph.actions
        assert "/controller_manager/switch_controller" in graph.services
        assert not any("/_action/" in s for s in graph.services + graph.topics)
        assert not any(s.endswith("/get_parameters") for s in graph.services)


async def test_progress_follows_controller_feedback(robot: FakeRosRobot) -> None:
    async with running_backend(ros_policy()) as b:
        plan = await b.plan_to_joints(target(0.5), 0.2, 0.2)
        seen: list[tuple[float, float]] = []
        t0 = time.monotonic()

        async def on_progress(fraction: float, message: str) -> None:
            seen.append((time.monotonic() - t0, fraction))

        res = await b.execute(plan, on_progress, lambda: False)
        assert res.success
        fractions = [f for _, f in seen]
        assert fractions == sorted(fractions) and fractions[-1] == 1.0
        assert any(0.3 < f < 0.9 for f in fractions)
        assert not math.isnan(sum(fractions))
