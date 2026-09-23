"""The real CLI (`armguard-mcp --backend ros2`) as a stdio subprocess, driving the fake FR3 cell."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("rclpy", reason="needs a sourced ROS 2 environment (rclpy)")

import yaml
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from tests.conftest import FR3_POLICY, text

from armguard_mcp.kinematics import FR3_READY
from tests_ros.fake_robot import FakeRosRobot

pytestmark = pytest.mark.anyio


async def test_cli_stdio_with_ros2_backend(robot: FakeRosRobot, tmp_path) -> None:
    cfg = tmp_path / "ros2.yaml"
    cfg.write_text(yaml.safe_dump({"ros2": {"startup_timeout_s": 10.0}}))
    # Same slack as tests_ros/conftest.ros_policy: the fake robot's wrench can stall on slow CI runners.
    policy_dict = yaml.safe_load(FR3_POLICY.read_text())
    policy_dict["force"]["wrench_timeout_s"] = 0.5
    policy_file = tmp_path / "fr3.yaml"
    policy_file.write_text(yaml.safe_dump(policy_dict))
    audit = tmp_path / "audit.jsonl"
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "armguard_mcp",
            "--policy",
            str(policy_file),
            "--backend",
            "ros2",
            "--ros2-config",
            str(cfg),
            "--audit-log",
            str(audit),
        ],
        env=dict(os.environ),  # ROS paths, ROS_DOMAIN_ID and discovery range from conftest
    )
    goal = list(FR3_READY)
    goal[6] += 0.2  # small wrist rotation: inside the envelope, no approval needed
    async with Client(params) as c:
        st = await c.call_tool("get_robot_state", {})
        assert not st.is_error, text(st)
        assert st.structured_content["backend"] == "ros2"
        p = (await c.call_tool("plan_to_joints", {"joint_positions": goal})).structured_content
        assert p["status"] == "executable", p
        r = await c.call_tool("execute_plan", {"plan_id": p["plan_id"]})
        assert not r.is_error, text(r)
        assert r.structured_content["status"] == "completed"
    assert robot.joints() == pytest.approx(goal, abs=1e-6)
    assert '"tool":"execute_plan"' in audit.read_text()
