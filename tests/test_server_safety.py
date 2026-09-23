"""E-stop, rate limits, tool groups, gripper/controller/perception guards and audit, end to end."""

from __future__ import annotations

import base64
import json

import pytest
from mcp.client import Client
from mcp.types import ImageContent

from armguard_mcp.imaging import pillow_available, png_size
from armguard_mcp.policy import TOOL_GROUPS
from tests.conftest import (
    READONLY_POLICY,
    READY,
    Elicitor,
    FakeClock,
    audit_events,
    make_app,
    make_policy,
    text,
)

pytestmark = pytest.mark.anyio


def target(j1: float) -> list[float]:
    q = list(READY)
    q[0] = j1
    return q


# --- e-stop ------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_estop_blocks_actuation_invalidates_plans_and_reset_needs_approval(mode: str) -> None:
    app = make_app(make_policy(approval={"mode": "never"}))  # 'never' does not apply to reset_estop
    human = Elicitor("decline")
    async with Client(app.server, mode=mode, elicitation_callback=human) as c:
        plan = (await c.call_tool("plan_to_joints", {"joint_positions": target(0.3)})).structured_content
        r = await c.call_tool("estop", {"reason": "person entered the cell"})
        assert not r.is_error and r.structured_content["estopped"]
        assert r.structured_content["reason"] == "person entered the cell"

        for name, args in [
            ("execute_plan", {"plan_id": plan["plan_id"]}),
            ("plan_to_joints", {"joint_positions": target(0.2)}),
            ("gripper_move", {"width_m": 0.04}),
            ("gripper_grasp", {"width_m": 0.03, "force_n": 10.0}),
            ("switch_controllers", {"activate": ["joint_impedance_controller"]}),
            ("set_collision_thresholds", {"force_n": 20.0, "torque_nm": 4.0}),
            ("error_recovery", {}),
        ]:
            res = await c.call_tool(name, args)
            assert res.is_error and "e-stop" in text(res), (name, text(res))
        assert human.messages == []  # nothing actuating even got to ask a human

        # observation and stop tools keep working
        assert not (await c.call_tool("get_robot_state", {})).is_error
        assert not (await c.call_tool("stop_motion", {})).is_error

        # reset: human declines -> still e-stopped
        r = await c.call_tool("reset_estop", {})
        assert r.is_error and "declined" in text(r)
        assert len(human.messages) == 1 and "person entered the cell" in human.messages[0]
        assert (await c.call_tool("get_safety_status", {})).structured_content["estopped"]

        # reset: human approves
        human.action = "accept"
        r = await c.call_tool("reset_estop", {})
        assert not r.is_error and not r.structured_content["estopped"]

        # the pre-e-stop plan stays invalid
        r = await c.call_tool("execute_plan", {"plan_id": plan["plan_id"]})
        assert r.is_error and "invalidated: e-stop" in text(r)
    events = [e["event"] for e in app.guard.audit.tail(200)]
    assert "estop" in events and "estop_reset" in events


async def test_reset_estop_denied_for_client_without_elicitation() -> None:
    app = make_app(make_policy(approval={"mode": "never"}))
    async with Client(app.server) as c:
        await c.call_tool("estop", {})
        r = await c.call_tool("reset_estop", {})
        assert r.is_error and "does not support elicitation" in text(r)
        assert (await c.call_tool("get_safety_status", {})).structured_content["estopped"]


async def test_reset_estop_when_not_estopped_is_a_noop_without_prompt() -> None:
    app = make_app()
    human = Elicitor()
    async with Client(app.server, elicitation_callback=human) as c:
        r = await c.call_tool("reset_estop", {})
        assert not r.is_error and not r.structured_content["estopped"]
    assert human.messages == []


async def test_force_violation_reset_flow() -> None:
    app = make_app(make_policy(approval={"mode": "never"}), speedup=10.0, table_height=0.40)
    human = Elicitor()
    async with Client(app.server, elicitation_callback=human) as c:
        r = await c.call_tool(
            "plan_cartesian_path", {"waypoints": [{"position": {"x": 0.3069, "y": 0, "z": 0.3}}]}
        )
        r = await c.call_tool("execute_plan", {"plan_id": r.structured_content["plan_id"]})
        assert r.is_error
        r = await c.call_tool("reset_estop", {})
        assert not r.is_error
        assert "FORCE-LIMIT VIOLATION" in human.messages[0]
        st = r.structured_content
        assert not st["estopped"] and not st["force_violation_latched"]
        # still in contact: pushing further down is aborted again (force rises above its starting level)
        z = (await c.call_tool("get_robot_state", {})).structured_content["ee_pose"]["position"]["z"]
        r = await c.call_tool(
            "plan_cartesian_path", {"waypoints": [{"position": {"x": 0.3069, "y": 0, "z": z - 0.02}}]}
        )
        r = await c.call_tool("execute_plan", {"plan_id": r.structured_content["plan_id"]})
        assert r.is_error and "ABORTED" in text(r)
        assert not (await c.call_tool("reset_estop", {})).is_error
        # retreat upwards is possible again after the reset
        r = await c.call_tool(
            "plan_cartesian_path", {"waypoints": [{"position": {"x": 0.3069, "y": 0, "z": 0.45}}]}
        )
        assert r.structured_content["status"] == "executable"
        assert not (await c.call_tool("execute_plan", {"plan_id": r.structured_content["plan_id"]})).is_error


# --- rate limits -------------------------------------------------------------------------
async def test_rate_limit_exceeded_but_safety_tools_never_limited() -> None:
    clk = FakeClock(0.0)
    policy = make_policy(rate_limits={"global_per_minute": 4, "default_per_minute": 4, "per_tool": {}})
    app = make_app(policy, monotonic=clk)
    async with Client(app.server) as c:
        for _ in range(4):
            assert not (await c.call_tool("get_robot_state", {})).is_error
        r = await c.call_tool("get_robot_state", {})
        assert r.is_error and "rate limit exceeded" in text(r)
        for _ in range(10):
            assert not (await c.call_tool("get_safety_status", {})).is_error
            assert not (await c.call_tool("stop_motion", {})).is_error
        assert not (await c.call_tool("estop", {"reason": "test"})).is_error
        clk.advance(30.0)  # refills 2 tokens
        assert not (await c.call_tool("get_robot_state", {})).is_error
    denied = [e for e in audit_events(app, "get_robot_state") if e["outcome"] == "denied"]
    assert denied and "rate limit" in denied[0]["error"]


async def test_per_tool_rate_limit_on_execute() -> None:
    policy = make_policy(approval={"mode": "never"}, rate_limits={"per_tool": {"execute_plan": 1}})
    app = make_app(policy)
    async with Client(app.server) as c:
        p1 = (await c.call_tool("plan_to_joints", {"joint_positions": target(0.1)})).structured_content
        assert not (await c.call_tool("execute_plan", {"plan_id": p1["plan_id"]})).is_error
        p2 = (await c.call_tool("plan_to_joints", {"joint_positions": target(0.2)})).structured_content
        r = await c.call_tool("execute_plan", {"plan_id": p2["plan_id"]})
        assert r.is_error and "rate limit" in text(r)


# --- tool groups ---------------------------------------------------------------------------
async def test_readonly_policy_lists_only_enabled_groups() -> None:
    from armguard_mcp.policy import load_policy

    app = make_app(load_policy(READONLY_POLICY))
    async with Client(app.server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        r = await c.call_tool("plan_to_joints", {"joint_positions": READY})
        assert r.is_error
    expected = set(TOOL_GROUPS["introspect"]) | set(TOOL_GROUPS["perception"]) | set(TOOL_GROUPS["safety"])
    assert names == expected
    assert "execute_plan" not in names and "gripper_grasp" not in names and "stop_motion" in names


async def test_tool_annotations_and_hidden_approval_parameter() -> None:
    app = make_app()
    async with Client(app.server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert len(tools) == sum(len(v) for v in TOOL_GROUPS.values())
    ex = tools["execute_plan"]
    assert ex.annotations.destructive_hint is True and ex.annotations.read_only_hint is False
    assert set(ex.input_schema["properties"]) == {"plan_id"}  # approval is resolved server-side
    assert "approval" not in tools["reset_estop"].input_schema.get("properties", {})
    assert tools["get_robot_state"].annotations.read_only_hint is True
    assert tools["plan_to_joints"].annotations.read_only_hint is True
    assert "server-side" in tools["get_safety_envelope"].description
    assert "rad" in tools["plan_to_joints"].description


# --- gripper -------------------------------------------------------------------------------
async def test_gripper_limits() -> None:
    app = make_app(object_width=0.03)
    async with Client(app.server) as c:
        r = await c.call_tool("gripper_grasp", {"width_m": 0.03, "force_n": 80.0})
        assert r.is_error and "force_n=80.0 N rejected" in text(r) and "40.0" in text(r)
        r = await c.call_tool("gripper_move", {"width_m": 0.2})
        assert r.is_error and "outside the allowed range" in text(r)
        r = await c.call_tool("gripper_move", {"width_m": 0.05, "speed_mps": 5.0})
        assert r.is_error and "speed_mps" in text(r)
        r = await c.call_tool("gripper_grasp", {"width_m": 0.03, "force_n": 20.0})
        assert not r.is_error and r.structured_content["ok"] and r.structured_content["data"]["is_grasped"]
        r = await c.call_tool("gripper_home", {})
        assert not r.is_error and r.structured_content["data"]["width_m"] == pytest.approx(0.08)
        r = await c.call_tool("gripper_grasp", {"width_m": 0.06, "force_n": 20.0})  # wrong expected width
        assert not r.is_error and not r.structured_content["ok"]


# --- controllers ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_controller_switch_allowlist_and_approval(mode: str) -> None:
    policy = make_policy(controllers={"allowlist": ["fr3_arm_controller", "cartesian_impedance_controller"]})
    app = make_app(policy)
    human = Elicitor()
    async with Client(app.server, mode=mode, elicitation_callback=human) as c:
        r = await c.call_tool("switch_controllers", {"activate": ["joint_state_broadcaster"]})
        assert r.is_error and "not in the policy allowlist" in text(r)
        r = await c.call_tool("switch_controllers", {"deactivate": ["joint_impedance_controller"]})
        assert r.is_error and "not in the policy allowlist" in text(r)
        assert human.messages == []
        r = await c.call_tool(
            "switch_controllers",
            {"activate": ["cartesian_impedance_controller"], "deactivate": ["fr3_arm_controller"]},
        )
        assert not r.is_error, text(r)
        states = {x["name"]: x["state"] for x in r.structured_content["controllers"]}
        assert (
            states["cartesian_impedance_controller"] == "active"
            and states["fr3_arm_controller"] == "inactive"
        )
        assert len(human.messages) == 1 and "CONTROLLER SWITCH" in human.messages[0]
        # trajectory execution now fails cleanly because the trajectory controller is inactive
        p = (await c.call_tool("plan_to_joints", {"joint_positions": target(0.1)})).structured_content
        r = await c.call_tool("execute_plan", {"plan_id": p["plan_id"]})
        assert r.is_error and "not active" in text(r)


async def test_collision_thresholds_capped_and_approved() -> None:
    app = make_app()
    human = Elicitor()
    async with Client(app.server, elicitation_callback=human) as c:
        r = await c.call_tool("set_collision_thresholds", {"force_n": 100.0, "torque_nm": 4.0})
        assert r.is_error and "max_contact_force_n" in text(r)
        assert human.messages == []
        r = await c.call_tool("set_collision_thresholds", {"force_n": 20.0, "torque_nm": 4.0})
        assert not r.is_error, text(r)
        assert len(human.messages) == 1
    assert app.guard.backend._collision_force_n == 20.0


async def test_error_recovery_requires_approval() -> None:
    app = make_app()
    app.guard.backend._in_error = True
    async with Client(app.server, elicitation_callback=Elicitor("decline")) as c:
        assert (await c.call_tool("error_recovery", {})).is_error
    assert app.guard.backend._in_error
    async with Client(app.server, elicitation_callback=Elicitor("accept")) as c:
        assert not (await c.call_tool("error_recovery", {})).is_error
    assert not app.guard.backend._in_error


# --- perception + audit --------------------------------------------------------------------
async def test_camera_snapshot_allowlist_and_audit_redaction(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    app = make_app(audit_path=audit_path)
    async with Client(app.server) as c:
        r = await c.call_tool("camera_snapshot", {"topic": "/camera/color/image_raw"})
        assert not r.is_error, text(r)
        img = next(x for x in r.content if isinstance(x, ImageContent))
        data = base64.b64decode(img.data)
        assert img.mime_type == "image/png" and png_size(data) == (64, 48)
        meta = json.loads(next(x.text for x in r.content if x.type == "text"))
        assert meta["width"] == 64 and meta["topic"] == "/camera/color/image_raw"
        r = await c.call_tool("camera_snapshot", {"topic": "/secret/camera"})
        assert r.is_error and "not allowlisted" in text(r)
    app.guard.audit.close()
    raw = audit_path.read_text()
    assert "iVBOR" not in raw and img.data[:40] not in raw  # no image payload in the audit log
    records = [json.loads(line) for line in raw.splitlines()]
    snaps = [x for x in records if x.get("tool") == "camera_snapshot"]
    assert [x["outcome"] for x in snaps] == ["ok", "denied"]
    assert snaps[0]["image"]["width"] == 64 and snaps[0]["image"]["size_bytes"] == len(data)
    assert records[0]["event"] == "server_start"


async def test_camera_snapshot_large_image_downscaled_or_refused() -> None:
    app = make_app(image_size=(640, 480))  # policy max_image_width = 320
    async with Client(app.server) as c:
        r = await c.call_tool("camera_snapshot", {"topic": "/camera/color/image_raw"})
    if pillow_available():
        assert not r.is_error
        img = next(x for x in r.content if isinstance(x, ImageContent))
        assert png_size(base64.b64decode(img.data)) == (320, 240)
    else:
        assert r.is_error and "Pillow" in text(r)


async def test_introspection_tools() -> None:
    app = make_app()
    async with Client(app.server) as c:
        env = (await c.call_tool("get_safety_envelope", {})).structured_content
        assert "fr3_joint4: [-3.0421, -0.1518] rad" in env["summary"]
        assert env["workspace"]["keep_out"][0]["name"] == "camera_mount"
        tf = await c.call_tool(
            "lookup_transform", {"target_frame": "fr3_link0", "source_frame": "fr3_hand_tcp"}
        )
        assert tf.structured_content["position"]["x"] == pytest.approx(0.3069, abs=1e-3)
        r = await c.call_tool("lookup_transform", {"target_frame": "fr3_link0", "source_frame": "moon"})
        assert r.is_error and "unknown frame" in text(r)
        ctrl = (await c.call_tool("list_controllers", {})).structured_content
        assert {
            "name": "fr3_arm_controller",
            "type": "joint_trajectory_controller/JointTrajectoryController",
            "state": "active",
        } in ctrl["controllers"]
        graph = (await c.call_tool("list_ros_graph", {})).structured_content
        assert "/fr3_arm_controller/follow_joint_trajectory" in graph["actions"]
        tail = (await c.call_tool("get_audit_tail", {"n": 2})).structured_content["events"]
        assert len(tail) == 2 and tail[-1]["tool"] == "list_ros_graph"
        assert (await c.call_tool("get_audit_tail", {"n": 0})).is_error  # validated: 1..200
