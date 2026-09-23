"""Unit tests for the pure safety layer: envelope, rate limiter, audit log, plan store."""

from __future__ import annotations

import itertools
import json
import math

import pytest

from armguard_mcp.kinematics import FR3Kinematics
from armguard_mcp.models import EnvelopeVerdict, Plan, PlanSummary, Pose, Quaternion, Vector3, Wrench
from armguard_mcp.plans import PlanAlreadyUsed, PlanExpired, PlanInvalidated, PlanNotFound, PlanStore
from armguard_mcp.policy import RateLimitsSection
from armguard_mcp.safety.audit import AuditLogger, redact
from armguard_mcp.safety.envelope import (
    check_joint_positions,
    check_plan,
    check_tcp_position,
    check_wrench,
    clamp_scaling,
    densify,
    max_joint_velocity_ratio,
)
from armguard_mcp.safety.ratelimit import RateLimiter, RateLimitExceeded
from tests.conftest import READY, FakeClock, make_policy

KIN = FR3Kinematics()


def mk_plan(
    waypoints: list[list[float]], times: list[float] | None = None, kind: str = "joints", **kw
) -> Plan:
    times = times or [i * 1.0 for i in range(len(waypoints))]
    return Plan(
        plan_id="p",
        kind=kind,  # type: ignore[arg-type]
        joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
        waypoints=waypoints,
        time_from_start=times,
        duration_s=times[-1],
        created_at=0.0,
        velocity_scaling=kw.get("vs", 0.1),
        acceleration_scaling=kw.get("as_", 0.1),
    )


def verdict_for(plan: Plan, policy=None) -> EnvelopeVerdict:
    policy = policy or make_policy()
    samples = densify(plan, policy.motion.check_resolution_rad)
    tcp = [KIN.fk(q)[i][3] for q in samples for i in range(3)]
    tcp3 = [tuple(tcp[i : i + 3]) for i in range(0, len(tcp), 3)]
    return check_plan(plan, policy, samples, tcp3)


def codes(v) -> set[str]:
    items = v.violations if isinstance(v, EnvelopeVerdict) else v
    return {x.code for x in items}


# --- envelope --------------------------------------------------------------------------
def test_joint_limits_hard_and_margin_soft() -> None:
    p = make_policy()
    assert check_joint_positions(READY, p) == []
    q = list(READY)
    q[3] = -0.05  # joint4 max is -0.1518
    (v,) = check_joint_positions(q, p)
    assert v.code == "JOINT_LIMIT" and v.severity == "hard" and v.detail["joint"] == "fr3_joint4"
    q[3] = -0.1518 - 0.01  # within 0.05 rad margin
    (v,) = check_joint_positions(q, p)
    assert v.code == "NEAR_JOINT_LIMIT" and v.severity == "soft"
    assert codes(check_joint_positions([0.0] * 3, p)) == {"JOINT_COUNT"}
    assert codes(check_joint_positions([math.nan, *READY[1:]], p)) == {"NOT_FINITE"}


def test_workspace_and_keep_out() -> None:
    p = make_policy()
    assert check_tcp_position((0.4, 0.0, 0.4), p) == []
    assert codes(check_tcp_position((0.4, 0.0, 0.0), p)) == {"WORKSPACE"}  # below z min
    assert codes(check_tcp_position((1.2, 0.0, 0.4), p)) == {"WORKSPACE"}
    vs = check_tcp_position((0.6, 0.4, 0.3), p)
    assert codes(vs) == {"KEEP_OUT"} and vs[0].detail["zone"] == "camera_mount"


def test_plan_inside_envelope() -> None:
    q1 = list(READY)
    q1[0] = 0.3
    v = verdict_for(mk_plan([READY, q1], [0.0, 3.0]))
    assert v.ok and v.inside_envelope, v


def test_plan_joint_step_hard_and_soft() -> None:
    q1 = list(READY)
    q1[0] = 1.0  # 1.0 rad travel > soft 0.8
    v = verdict_for(mk_plan([READY, q1], [0.0, 10.0]))
    assert v.ok and not v.inside_envelope and codes(v) == {"LARGE_MOTION"}
    # Back-and-forth motion counts total travel: 0.9 + 0.9 = 1.8 rad > hard 1.6
    q2 = list(READY)
    q2[6] = READY[6] + 0.9
    v = verdict_for(mk_plan([READY, q2, READY], [0.0, 10.0, 20.0]))
    assert not v.ok and "STEP_TOO_LARGE" in codes(v)


def test_plan_workspace_violation_mid_path_is_detected() -> None:
    q1 = list(READY)
    q1[0] = 1.5  # sweeps the TCP out of the workspace box (x < 0.15)
    v = verdict_for(mk_plan([READY, q1], [0.0, 30.0]), make_policy(motion={"max_joint_step_rad": 2.0}))
    assert not v.ok and "WORKSPACE" in codes(v)


def test_plan_keep_out_violation() -> None:
    p = make_policy(
        workspace={"keep_out": [{"name": "fixture", "min": [0.25, -0.1, 0.4], "max": [0.35, 0.1, 0.6]}]}
    )
    q1 = list(READY)
    q1[0] = 0.2
    v = verdict_for(mk_plan([READY, q1], [0.0, 3.0]), p)
    assert not v.ok and codes(v) == {"KEEP_OUT"}


def test_plan_joint_limit_violation_along_path() -> None:
    q1 = list(READY)
    q1[5] = 4.6  # joint6 max 4.5169
    v = verdict_for(mk_plan([READY, q1], [0.0, 30.0]), make_policy(motion={"max_joint_step_rad": 3.5}))
    assert not v.ok and "JOINT_LIMIT" in codes(v)


def test_plan_velocity_cap() -> None:
    q1 = list(READY)
    q1[0] = 0.5
    plan = mk_plan([READY, q1], [0.0, 0.5])  # 1 rad/s = 0.38 of the 2.62 rad/s limit > 0.3 cap
    assert max_joint_velocity_ratio(plan, make_policy()) == pytest.approx(1.0 / 2.62)
    v = verdict_for(plan)
    assert not v.ok and "VELOCITY" in codes(v)


def test_plan_malformed() -> None:
    q1 = list(READY)
    v = verdict_for(mk_plan([READY, q1], [0.0, 0.0]))
    assert codes(v) == {"MALFORMED"}


def test_cartesian_length_cap() -> None:
    p = make_policy(motion={"max_cartesian_step_m": 0.05})
    q1 = list(READY)
    q1[0] = 0.3  # TCP moves ~0.09 m
    v = verdict_for(mk_plan([READY, q1], [0.0, 5.0], kind="cartesian"), p)
    assert "CARTESIAN_STEP_TOO_LARGE" in codes(v)


def test_densify_resolution() -> None:
    q1 = list(READY)
    q1[0] = 0.1
    s = densify(mk_plan([READY, q1]), 0.02)
    assert len(s) == 6 and s[-1] == q1
    assert max(abs(b[0] - a[0]) for a, b in itertools.pairwise(s)) <= 0.02 + 1e-12


def test_clamp_scaling() -> None:
    assert clamp_scaling(None, 0.1, 0.3) == (0.1, None)
    assert clamp_scaling(0.2, 0.1, 0.3) == (0.2, None)
    val, note = clamp_scaling(0.9, 0.1, 0.3)
    assert val == 0.3 and "clamped" in note
    with pytest.raises(ValueError):
        clamp_scaling(-1.0, 0.1, 0.3)


def test_check_wrench() -> None:
    p = make_policy()
    ok = Wrench(frame_id="b", force=Vector3(x=0, y=3, z=4), torque=Vector3(x=0, y=0, z=1))
    assert check_wrench(ok, p) == []
    bad = Wrench(frame_id="b", force=Vector3(x=0, y=0, z=30), torque=Vector3(x=0, y=0, z=6))
    assert codes(check_wrench(bad, p)) == {"FORCE_LIMIT", "TORQUE_LIMIT"}


# --- rate limiter ----------------------------------------------------------------------
def test_rate_limiter_per_tool_global_and_refill() -> None:
    clk = FakeClock(0.0)
    rl = RateLimiter(
        RateLimitsSection(global_per_minute=5, default_per_minute=3, per_tool={"execute_plan": 1}), clk
    )
    rl.acquire("execute_plan")
    assert not rl.would_allow("execute_plan")
    with pytest.raises(RateLimitExceeded, match="per-tool"):
        rl.acquire("execute_plan")
    for _ in range(3):
        rl.acquire("get_robot_state")
    with pytest.raises(RateLimitExceeded, match="per-tool"):
        rl.acquire("get_robot_state")
    rl.acquire("list_controllers")  # 5th global token
    with pytest.raises(RateLimitExceeded, match="global"):
        rl.acquire("lookup_transform")
    # safety tools are never limited
    for _ in range(100):
        rl.acquire("estop")
        rl.acquire("stop_motion")
        rl.acquire("get_safety_status")
    clk.advance(60.0)
    rl.acquire("execute_plan")


# --- audit -----------------------------------------------------------------------------
def test_audit_jsonl_and_redaction(tmp_path) -> None:
    path = tmp_path / "logs" / "audit.jsonl"
    a = AuditLogger(path, ring_size=3)
    a.log(
        "tool_call",
        tool="camera_snapshot",
        args={"topic": "/cam"},
        data="iVBORw0KGgo" * 50,
        raw=b"\x89PNG....",
    )
    a.log("tool_call", tool="x", args={"image": "AAAA", "nested": {"png": "BBBB", "ok": 1}})
    for i in range(3):
        a.log("tick", n=i)
    a.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 5 and [r["seq"] for r in lines] == [1, 2, 3, 4, 5]
    assert lines[0]["data"].startswith("<redacted") and lines[0]["raw"] == "<redacted 8 bytes>"
    assert lines[1]["args"]["image"].startswith("<redacted") and lines[1]["args"]["nested"]["png"].startswith(
        "<redacted"
    )
    assert lines[1]["args"]["nested"]["ok"] == 1
    assert "iVBOR" not in path.read_text()
    assert lines[0]["ts"].endswith("Z")
    assert [e["n"] for e in a.tail(10)] == [0, 1, 2]  # ring buffer of 3


def test_redact_truncates_long_strings() -> None:
    out = redact({"msg": "x" * 5000})
    assert len(out["msg"]) < 2200 and "truncated" in out["msg"]


# --- plan store ------------------------------------------------------------------------
def _summary(plan: Plan) -> PlanSummary:
    return PlanSummary(
        plan_id=plan.plan_id,
        kind=plan.kind,
        status="executable",
        executable=True,
        duration_s=plan.duration_s,
        num_waypoints=len(plan.waypoints),
        start_joint_positions=plan.start,
        final_joint_positions=plan.final,
        final_ee_pose=Pose(frame_id="b", position=Vector3(x=0, y=0, z=0), orientation=Quaternion()),
        max_joint_velocity_ratio=0.1,
        max_joint_travel_rad=0.1,
        tcp_path_length_m=0.1,
        velocity_scaling=0.1,
        acceleration_scaling=0.1,
        requires_approval=False,
        dry_run=False,
    )


def test_plan_store_ttl_single_use_invalidate() -> None:
    clk = FakeClock(100.0)
    store = PlanStore(ttl_s=10.0, clock=clk)
    ok = EnvelopeVerdict.from_violations([])
    for pid in ("a", "b", "c"):
        plan = mk_plan([READY, READY]).model_copy(update={"plan_id": pid})
        store.put(plan, _summary(plan), ok)
    assert store.peek("a").plan.plan_id == "a"
    store.consume("a")
    with pytest.raises(PlanAlreadyUsed):
        store.peek("a")
    with pytest.raises(PlanNotFound):
        store.peek("zzz")
    clk.advance(10.0)
    with pytest.raises(PlanExpired):
        store.peek("b")
    plan = mk_plan([READY, READY]).model_copy(update={"plan_id": "d"})
    store.put(plan, _summary(plan), ok)
    # storing "d" purged the expired "c", so only "d" is outstanding
    assert store.invalidate_all("e-stop") == 1
    with pytest.raises(PlanInvalidated, match="e-stop"):
        store.peek("d")
    with pytest.raises(PlanExpired):
        store.peek("c")
