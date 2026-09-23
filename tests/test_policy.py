from __future__ import annotations

import pytest

from armguard_mcp.policy import TOOL_GROUPS, Policy, PolicyError, load_policy
from tests.conftest import FR3_POLICY, READONLY_POLICY, deep_merge, fr3_dict, make_policy


def test_example_policies_load() -> None:
    fr3 = load_policy(FR3_POLICY)
    assert fr3.robot.joint_names == [f"fr3_joint{i}" for i in range(1, 8)]
    assert fr3.robot.joint_limits["fr3_joint4"].max == -0.1518
    assert fr3.robot.joint_limits["fr3_joint6"].min == 0.5445
    assert fr3.robot.base_frame == "fr3_link0" and fr3.robot.ee_frame == "fr3_hand_tcp"
    ro = load_policy(READONLY_POLICY)
    assert set(ro.tools.enabled) == {"introspect", "perception", "safety"}


def test_safety_group_cannot_be_disabled() -> None:
    p = make_policy(tools={"enabled": ["introspect"]})
    assert "safety" in p.tools.enabled
    assert set(TOOL_GROUPS["safety"]) <= set(p.enabled_tools())


def test_reset_estop_always_requires_approval() -> None:
    p = make_policy(approval={"mode": "never", "require_for": ["execute"]})
    assert "reset_estop" in p.approval.require_for


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(PolicyError, match=r"motion\.max_speed"):
        make_policy(motion={"max_speed": 3.0})
    with pytest.raises(PolicyError, match="Extra inputs are not permitted"):
        Policy.from_dict(deep_merge(fr3_dict(), {"surprise": 1}))


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"motion": {"max_velocity_scaling": 1.5}}, "max_velocity_scaling"),
        ({"motion": {"max_velocity_scaling": 0.0}}, "max_velocity_scaling"),
        ({"motion": {"default_velocity_scaling": 0.5}}, "default_velocity_scaling"),
        ({"robot": {"home_joint_positions": [0, 0, 0, 0, 0, 0, 0]}}, "home position of fr3_joint4"),
        (
            {"robot": {"joint_limits": {"fr3_joint1": {"min": 1.0, "max": -1.0, "max_velocity": 1}}}},
            "must be < max",
        ),
        ({"workspace": {"box": {"min": [1, 0, 0], "max": [0, 1, 1]}}}, "box min"),
        ({"rate_limits": {"per_tool": {"launch_missiles": 1}}}, "unknown tools"),
        ({"approval": {"mode": "sometimes"}}, "approval.mode"),
        ({"gripper": {"min_width_m": 0.1, "max_width_m": 0.08}}, "min_width_m"),
        ({"tools": {"enabled": ["introspect", "teleport"]}}, "tools.enabled"),
    ],
)
def test_invalid_values(override: dict, pattern: str) -> None:
    with pytest.raises(PolicyError, match=pattern):
        make_policy(**override)


def test_joint_limits_must_match_joint_names() -> None:
    d = fr3_dict()
    del d["robot"]["joint_limits"]["fr3_joint7"]
    with pytest.raises(PolicyError, match="missing=\\['fr3_joint7'\\]"):
        Policy.from_dict(d)


def test_gripper_group_requires_gripper_section() -> None:
    d = fr3_dict()
    del d["gripper"]
    with pytest.raises(PolicyError, match="no gripper section"):
        Policy.from_dict(d)


def test_file_errors(tmp_path) -> None:
    with pytest.raises(PolicyError, match="cannot read"):
        load_policy(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("robot: [unclosed")
    with pytest.raises(PolicyError, match="YAML parse error"):
        load_policy(bad)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("42")
    with pytest.raises(PolicyError, match="must be a YAML mapping"):
        load_policy(scalar)


def test_policy_is_immutable() -> None:
    p = make_policy()
    with pytest.raises(Exception, match="frozen"):
        p.dry_run = True  # type: ignore[misc]
    assert p.with_dry_run().dry_run is True and p.dry_run is False
