"""End to end: MCP client -> armguard server -> Ros2Backend -> rclpy -> fake FR3 cell."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest

pytest.importorskip("rclpy", reason="needs a sourced ROS 2 environment (rclpy)")

from mcp.client import Client
from tests.conftest import Elicitor, text

from armguard_mcp.backends.ros2 import Ros2Backend
from armguard_mcp.kinematics import FR3_READY
from armguard_mcp.server import ArmGuardApp, build
from tests_ros.conftest import ros_policy
from tests_ros.fake_robot import HAVE_FRANKA_MSGS, FakeRosRobot

pytestmark = pytest.mark.anyio

READY = list(FR3_READY)
MODES = ["auto", "legacy"]


def target(j1: float) -> list[float]:
    q = list(READY)
    q[0] = j1
    return q


@asynccontextmanager
async def ros_app(policy: Any) -> AsyncIterator[ArmGuardApp]:
    backend = Ros2Backend.from_policy(policy)
    app = build(policy, backend)
    try:
        yield app
    finally:
        await backend.shutdown()


async def plan(c: Client, q: list[float], **kw: Any) -> dict[str, Any]:
    r = await c.call_tool("plan_to_joints", {"joint_positions": q, **kw})
    assert not r.is_error, text(r)
    return r.structured_content


async def test_get_robot_state(robot: FakeRosRobot) -> None:
    async with ros_app(ros_policy()) as app, Client(app.server) as c:
        r = await c.call_tool("get_robot_state", {})
        assert not r.is_error, text(r)
        s = r.structured_content
        assert s["backend"] == "ros2"
        assert s["joint_state"]["positions"] == pytest.approx(READY, abs=1e-9)
        p = s["ee_pose"]["position"]
        assert (p["x"], p["y"], p["z"]) == pytest.approx((0.30689, 0.0, 0.48688), abs=1e-4)
        assert s["wrench"]["force"]["z"] == pytest.approx(0.0)
        assert s["gripper"]["width_m"] == pytest.approx(0.08)
        assert not s["safety"]["estopped"]


@pytest.mark.parametrize("mode", MODES)
async def test_plan_execute_with_human_approval(robot: FakeRosRobot, mode: str) -> None:
    app_policy = ros_policy(approval={"mode": "always"})
    human = Elicitor("accept", approve=True, operator="lynn")
    progress: list[float] = []

    async def on_progress(p: float, total: float | None, message: str | None) -> None:
        progress.append(p)

    async with ros_app(app_policy) as app, Client(app.server, mode=mode, elicitation_callback=human) as c:
        s = await plan(c, target(0.3))
        assert s["status"] == "needs_approval" and s["executable"]
        req = robot.plan_requests[-1]
        assert req.max_velocity_scaling_factor == pytest.approx(0.1)  # policy default scaling
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]}, progress_callback=on_progress)
        assert not r.is_error, text(r)
        rep = r.structured_content
        assert rep["status"] == "completed" and rep["approval"] == "human"
        assert rep["final_joint_positions"] == pytest.approx(target(0.3), abs=1e-6)
    assert len(human.messages) == 1 and s["plan_id"] in human.messages[0]
    assert len(robot.goals) == 1  # the robot was commanded exactly once
    assert robot.joints() == pytest.approx(target(0.3), abs=1e-6)
    assert progress and progress[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("mode", MODES)
async def test_declined_approval_never_reaches_the_controller(robot: FakeRosRobot, mode: str) -> None:
    human = Elicitor("decline")
    async with (
        ros_app(ros_policy(approval={"mode": "always"})) as app,
        Client(app.server, mode=mode, elicitation_callback=human) as c,
    ):
        s = await plan(c, target(0.3))
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "declined" in text(r)
    assert robot.goals == []
    assert robot.joints() == pytest.approx(READY, abs=1e-9)


async def test_stop_motion_cancels_the_trajectory(robot: FakeRosRobot) -> None:
    async with ros_app(ros_policy(approval={"mode": "never"})) as app, Client(app.server) as c:
        s = await plan(c, target(0.8), velocity_scaling=0.05)
        assert s["duration_s"] > 3.0
        results: list[Any] = []

        async def run() -> None:
            results.append(await c.call_tool("execute_plan", {"plan_id": s["plan_id"]}))

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            with anyio.fail_after(5):
                while not (await c.call_tool("get_motion_status", {})).structured_content["executing"]:
                    await anyio.sleep(0.05)
            await anyio.sleep(0.5)
            r = await c.call_tool("stop_motion", {})
            assert not r.is_error, text(r)
        rep = results[0]
        assert not rep.is_error, text(rep)
        assert rep.structured_content["status"] == "aborted"
    assert robot.cancels == 1
    assert 0.0 < robot.joints()[0] < 0.8


async def test_force_limit_aborts_and_latches_estop(robot: FakeRosRobot) -> None:
    robot.force_spike = (0.3, 40.0)  # 40 N > the policy's 25 N, at 30 % of the motion
    async with ros_app(ros_policy(approval={"mode": "never"})) as app, Client(app.server) as c:
        s = await plan(c, target(0.6))
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "ABORTED" in text(r) and "25" in text(r)
        status = (await c.call_tool("get_safety_status", {})).structured_content
        assert status["estopped"] and status["force_violation_latched"]
        r = await c.call_tool("plan_to_joints", {"joint_positions": target(0.0)})
        assert r.is_error and "force-limit violation latched" in text(r) and "reset_estop" in text(r)
    assert robot.cancels == 1
    assert robot.joints()[0] < 0.6


async def test_execution_refused_without_wrench() -> None:
    robot = FakeRosRobot(publish_wrench=False)
    try:
        policy = ros_policy({"startup_timeout_s": 2.0}, approval={"mode": "never"})
        async with ros_app(policy) as app, Client(app.server) as c:
            s = await plan(c, target(0.2))
            r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
            assert r.is_error and "no fresh external wrench" in text(r)
        assert robot.goals == []
    finally:
        robot.close()


@pytest.mark.parametrize("mode", MODES)
async def test_switch_controllers_with_approval(robot: FakeRosRobot, mode: str) -> None:
    human = Elicitor("accept", approve=True)
    async with ros_app(ros_policy()) as app, Client(app.server, mode=mode, elicitation_callback=human) as c:
        r = await c.call_tool(
            "switch_controllers",
            {"activate": ["cartesian_impedance_controller"], "deactivate": ["fr3_arm_controller"]},
        )
        assert not r.is_error, text(r)
        lst = (await c.call_tool("list_controllers", {})).structured_content
        states = {x["name"]: x["state"] for x in lst["controllers"]}
        assert states["cartesian_impedance_controller"] == "active"
        assert states["fr3_arm_controller"] == "inactive"
        # a controller outside the policy allowlist never reaches controller_manager
        r = await c.call_tool("switch_controllers", {"activate": ["joint_state_broadcaster"]})
        assert r.is_error
    assert len(human.messages) == 1 and len(robot.switch_requests) == 1


async def test_cartesian_path_and_pose_goal(robot: FakeRosRobot) -> None:
    async with ros_app(ros_policy(approval={"mode": "never"})) as app, Client(app.server) as c:
        ee = (await c.call_tool("get_robot_state", {})).structured_content["ee_pose"]["position"]
        r = await c.call_tool(
            "plan_cartesian_path",
            {"waypoints": [{"position": {"x": ee["x"], "y": ee["y"], "z": ee["z"] - 0.05}}]},
        )
        assert not r.is_error, text(r)
        s = r.structured_content
        assert s["kind"] == "cartesian" and s["tcp_path_length_m"] == pytest.approx(0.05, abs=2e-3)
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert not r.is_error, text(r)
        z = r.structured_content["final_ee_pose"]["position"]["z"]
        assert z == pytest.approx(ee["z"] - 0.05, abs=2e-3)

        r = await c.call_tool(
            "plan_to_pose", {"position": {"x": ee["x"] + 0.05, "y": 0.05, "z": ee["z"] - 0.05}}
        )
        assert not r.is_error, text(r)
        assert r.structured_content["executable"]
        # a goal outside the workspace box is planned by MoveIt but rejected by the envelope
        r = await c.call_tool("plan_to_pose", {"position": {"x": 0.3, "y": 0.0, "z": 0.95}})
        assert r.is_error or r.structured_content["status"] == "rejected"


async def test_camera_snapshot(robot: FakeRosRobot) -> None:
    async with ros_app(ros_policy()) as app, Client(app.server) as c:
        r = await c.call_tool("camera_snapshot", {"topic": "/camera/color/image_raw"})
        assert not r.is_error, text(r)
        img = next(x for x in r.content if x.type == "image")
        assert img.mime_type == "image/png"
        assert '"width": 320' in text(r)


async def test_gripper_tools(robot: FakeRosRobot) -> None:
    async with ros_app(ros_policy()) as app, Client(app.server) as c:
        r = await c.call_tool("gripper_move", {"width_m": 0.04})
        assert not r.is_error, text(r)
        assert robot.gripper_width == pytest.approx(0.04)
        robot.object_width = 0.03
        r = await c.call_tool("gripper_grasp", {"width_m": 0.03, "force_n": 20.0})
        assert not r.is_error and r.structured_content["ok"], text(r)
        n = len(robot.gripper_goals)
        r = await c.call_tool("gripper_grasp", {"width_m": 0.03, "force_n": 60.0})
        assert r.is_error and "policy maximum" in text(r)  # above the policy's 40 N
        assert len(robot.gripper_goals) == n  # rejected before reaching ROS


@pytest.mark.skipif(not HAVE_FRANKA_MSGS, reason="franka_msgs not built")
async def test_collision_thresholds_and_error_recovery_with_approval(robot: FakeRosRobot) -> None:
    human = Elicitor("accept", approve=True)
    async with ros_app(ros_policy()) as app, Client(app.server, elicitation_callback=human) as c:
        r = await c.call_tool("set_collision_thresholds", {"force_n": 20.0, "torque_nm": 5.0})
        assert not r.is_error, text(r)
        assert list(robot.collision_requests[-1].upper_force_thresholds_nominal)[:3] == [20.0] * 3
        r = await c.call_tool("set_collision_thresholds", {"force_n": 200.0, "torque_nm": 5.0})
        assert r.is_error and len(robot.collision_requests) == 1
        r = await c.call_tool("error_recovery", {})
        assert not r.is_error, text(r)
    assert robot.recoveries == 1 and len(human.messages) == 2


async def test_introspection(robot: FakeRosRobot) -> None:
    async with ros_app(ros_policy()) as app, Client(app.server) as c:
        g = (await c.call_tool("list_ros_graph", {})).structured_content
        assert (
            "/fake_fr3_cell" in g["nodes"] and "/fr3_arm_controller/follow_joint_trajectory" in g["actions"]
        )
        tf = (
            await c.call_tool("lookup_transform", {"target_frame": "world", "source_frame": "fr3_hand_tcp"})
        ).structured_content
        assert tf["position"]["z"] == pytest.approx(0.48688, abs=1e-4)
        r = await c.call_tool("lookup_transform", {"target_frame": "world", "source_frame": "nope"})
        assert r.is_error and "no transform" in text(r)
