"""Against a REAL MoveIt 2 ``move_group`` (OMPL, KDL IK, time parameterisation) on ROS 2 Jazzy.

Uses the Panda from ``moveit_resources_panda_moveit_config`` (the FR3 has the same kinematic
structure, so the fake cell can play the robot with prefix ``panda``). Planning, FK and
Cartesian paths come from move_group; execution goes to the fake FollowJointTrajectory server.

Skipped unless move_group, moveit_configs_utils and the Panda config are installed:
``apt install ros-jazzy-moveit-ros-move-group ros-jazzy-moveit-planners-ompl
ros-jazzy-moveit-kinematics ros-jazzy-moveit-configs-utils ros-jazzy-moveit-resources-panda-moveit-config``.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import signal
import subprocess
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest

pytest.importorskip("rclpy", reason="needs a sourced ROS 2 environment (rclpy)")
pytest.importorskip("moveit_configs_utils", reason="needs ros-jazzy-moveit-configs-utils")

import yaml
from mcp.client import Client
from tests.conftest import Elicitor, fr3_dict, text

from armguard_mcp import geometry as g
from armguard_mcp.backends.ros2 import Ros2Backend
from armguard_mcp.kinematics import FR3_READY, FR3Kinematics
from armguard_mcp.policy import Policy
from armguard_mcp.server import ArmGuardApp, build
from tests_ros.fake_robot import FakeRosRobot

pytestmark = pytest.mark.anyio

PANDA_CONFIG = "moveit_resources_panda_moveit_config"
READY = list(FR3_READY)
JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
# moveit_resources Panda URDF position limits and joint_limits.yaml velocities
LIMITS = [
    (-2.8973, 2.8973, 2.175),
    (-1.7628, 1.7628, 2.175),
    (-2.8973, 2.8973, 2.175),
    (-3.0718, -0.0698, 2.175),
    (-2.8973, 2.8973, 2.61),
    (-0.0175, 3.7525, 2.61),
    (-2.8973, 2.8973, 2.61),
]
PANDA_KIN = FR3Kinematics(prefix="panda")


def _have_move_group() -> bool:
    try:
        from ament_index_python.packages import get_package_prefix, get_package_share_directory

        get_package_share_directory(PANDA_CONFIG)
        return (
            Path(get_package_prefix("moveit_ros_move_group")) / "lib/moveit_ros_move_group/move_group"
        ).exists()
    except Exception:
        return False


if not _have_move_group() or shutil.which("ros2") is None:
    pytest.skip("MoveIt move_group / Panda config not installed", allow_module_level=True)


def panda_policy(**over: Any) -> Policy:
    d = fr3_dict()
    d["robot"] = {
        "name": "panda",
        "planning_group": "panda_arm",
        "base_frame": "panda_link0",
        "ee_frame": "panda_hand",
        "joint_names": JOINTS,
        "joint_limits": {
            j: {"min": lo, "max": hi, "max_velocity": v, "max_acceleration": 5.0}
            for j, (lo, hi, v) in zip(JOINTS, LIMITS, strict=True)
        },
        "home_joint_positions": READY,
    }
    d["controllers"] = {"allowlist": ["panda_arm_controller"]}
    d["tools"] = {"enabled": ["introspect", "motion", "safety"]}
    d.pop("gripper", None)
    d["ros2"] = {
        "startup_timeout_s": 30.0,
        "planning_time_s": 5.0,
        "trajectory_action": "/panda_arm_controller/follow_joint_trajectory",
        "gripper_interface": "none",
        "collision_behavior_service": None,
        "error_recovery_action": None,
    }
    d["force"]["wrench_timeout_s"] = 0.5  # slack for the in-process fake wrench on slow CI runners
    d.update(over)
    return Policy.from_dict(d)


def _write_move_group_params(path: Path) -> None:
    from moveit_configs_utils import MoveItConfigsBuilder

    cfg = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = cfg.to_dict()
    params.update({"use_sim_time": False, "publish_robot_description_semantic": True})
    path.write_text(yaml.safe_dump({"move_group": {"ros__parameters": params}}))


async def _wait_until_move_group_is_ready(timeout: float) -> str | None:
    backend = Ros2Backend.from_policy(panda_policy())
    await backend.start()
    deadline = time.monotonic() + timeout
    error: str | None = "timeout"
    try:
        while time.monotonic() < deadline:
            try:
                await backend.forward_kinematics(READY)
                await backend.get_ee_pose()
                return None
            except Exception as e:  # not ready yet
                error = f"{type(e).__name__}: {e}"
                backend._fk_cache.clear()
                await anyio.sleep(0.5)
        return error
    finally:
        await backend.shutdown()


def _stop_process_group(proc: subprocess.Popen[bytes]) -> None:
    for sig, wait_s in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)
        try:
            proc.wait(timeout=wait_s)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)  # reap any child that outlived the wrapper
            return
        except subprocess.TimeoutExpired:
            continue


@pytest.fixture(scope="module")
def moveit_cell(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FakeRosRobot]:
    tmp = tmp_path_factory.mktemp("move_group")
    params = tmp / "move_group.yaml"
    _write_move_group_params(params)
    robot = FakeRosRobot(prefix="panda", ee_frame="panda_hand", serve_moveit=False)
    log = (tmp / "move_group.log").open("w")
    proc = subprocess.Popen(
        ["ros2", "run", "moveit_ros_move_group", "move_group", "--ros-args", "--params-file", str(params)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ),
        # `ros2 run` is a wrapper: terminating it alone orphans the real move_group, which then keeps
        # answering /plan_kinematic_path for later test modules. Own a process group and stop all of it.
        start_new_session=True,
    )
    try:
        time.sleep(1.0)
        if proc.poll() is not None:
            pytest.fail(f"move_group exited early; see {tmp / 'move_group.log'}")
        # Readiness probe: move_group advertises its services before its planning scene has the
        # robot state and tf (world -> panda_link0). Wait until /compute_fk actually answers.
        error = anyio.run(_wait_until_move_group_is_ready, 60.0)
        if error:
            pytest.fail(f"move_group not ready: {error}; see {tmp / 'move_group.log'}")
        yield robot
    finally:
        _stop_process_group(proc)
        log.close()
        robot.close()


@asynccontextmanager
async def app_for(policy: Policy) -> AsyncIterator[ArmGuardApp]:
    backend = Ros2Backend.from_policy(policy)
    await backend.start()
    missing = [k for k, ok in backend.availability.items() if not ok]
    assert not missing, f"not available: {missing}"
    try:
        yield build(policy, backend)
    finally:
        await backend.shutdown()


async def robot_state(c: Client) -> dict[str, Any]:
    r = await c.call_tool("get_robot_state", {})
    assert not r.is_error, text(r)
    return r.structured_content


def target(j1: float) -> list[float]:
    q = list(READY)
    q[0] = j1
    return q


async def test_real_compute_fk_matches_the_dh_model(moveit_cell: FakeRosRobot) -> None:
    async with app_for(panda_policy()) as app:
        backend = app.guard.backend
        for q in (READY, target(0.4), [0.2, -0.3, 0.1, -2.0, 0.3, 1.9, 0.5]):
            pose = await backend.forward_kinematics(q)
            m = PANDA_KIN.chain(q)["panda_hand"]
            assert pose.position.as_tuple() == pytest.approx(tuple(g.translation(m)), abs=1e-6)
            assert g.angle_between_quats(pose.orientation.as_tuple(), g.quat_from_matrix(m)) < 1e-5


async def test_plan_and_execute_with_real_moveit(moveit_cell: FakeRosRobot) -> None:
    human = Elicitor("accept", approve=True)
    goals_before = len(moveit_cell.goals)
    async with (
        app_for(panda_policy(approval={"mode": "always"})) as app,
        Client(app.server, elicitation_callback=human) as c,
    ):
        start = (await robot_state(c))["joint_state"]["positions"]
        goal = list(start)
        goal[0] = start[0] + 0.3 if start[0] < 1.0 else start[0] - 0.3
        r = await c.call_tool("plan_to_joints", {"joint_positions": goal})
        assert not r.is_error, text(r)
        s = r.structured_content
        assert s["executable"] and s["num_waypoints"] >= 2 and s["duration_s"] > 0
        # MoveIt honoured the policy's default 10 % velocity scaling
        assert s["max_joint_velocity_ratio"] <= 0.105
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert not r.is_error, text(r)
        assert r.structured_content["status"] == "completed"
    assert len(moveit_cell.goals) == goals_before + 1
    assert moveit_cell.joints() == pytest.approx(goal, abs=2e-3)
    assert len(human.messages) == 1


async def test_cartesian_path_with_real_moveit(moveit_cell: FakeRosRobot) -> None:
    async with app_for(panda_policy(approval={"mode": "never"})) as app, Client(app.server) as c:
        ee = (await robot_state(c))["ee_pose"]["position"]
        r = await c.call_tool(
            "plan_cartesian_path",
            {"waypoints": [{"position": {"x": ee["x"], "y": ee["y"], "z": ee["z"] - 0.05}}]},
        )
        assert not r.is_error, text(r)
        s = r.structured_content
        assert s["kind"] == "cartesian" and s["executable"], s
        assert s["tcp_path_length_m"] == pytest.approx(0.05, abs=2e-3)
        assert s["duration_s"] > 0
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert not r.is_error, text(r)
        z = r.structured_content["final_ee_pose"]["position"]["z"]
        assert z == pytest.approx(ee["z"] - 0.05, abs=2e-3)


async def test_pose_goal_with_real_moveit(moveit_cell: FakeRosRobot) -> None:
    async with app_for(panda_policy(approval={"mode": "never"})) as app, Client(app.server) as c:
        ee = (await robot_state(c))["ee_pose"]["position"]
        want = {"x": ee["x"], "y": ee["y"] + 0.05, "z": ee["z"]}
        r = await c.call_tool("plan_to_pose", {"position": want})
        assert not r.is_error, text(r)
        s = r.structured_content
        assert s["executable"], s
        p = s["final_ee_pose"]["position"]
        assert math.dist((p["x"], p["y"], p["z"]), (want["x"], want["y"], want["z"])) < 2e-3


async def test_envelope_rejects_a_valid_moveit_plan(moveit_cell: FakeRosRobot) -> None:
    """MoveIt happily plans a big swing; the server's envelope refuses to execute it."""
    goals_before = len(moveit_cell.goals)
    async with app_for(panda_policy(approval={"mode": "never"})) as app, Client(app.server) as c:
        start = (await robot_state(c))["joint_state"]["positions"]
        goal = list(start)
        goal[0] = -2.0 if start[0] > 0 else 2.0  # > max_joint_step_rad and TCP leaves the box
        r = await c.call_tool("plan_to_joints", {"joint_positions": goal})
        assert not r.is_error, text(r)
        s = r.structured_content
        assert s["status"] == "rejected" and not s["executable"]
        codes = {v["code"] for v in s["violations"]}
        assert codes & {"STEP_TOO_LARGE", "WORKSPACE", "KEEP_OUT"}, codes
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "hard safety limits" in text(r)
    assert len(moveit_cell.goals) == goals_before
