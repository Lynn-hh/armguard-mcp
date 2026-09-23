"""Regression tests for the findings of the adversarial review (safety bypasses, MCP correctness,
concurrency). Each test fails on the code before the fix.

Several tests use FakeBackend subclasses whose reads *yield* to the event loop, the way every
rclpy-backed call does; the plain fake never yields there, which is how these bugs hid.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

import anyio
import mcp.types as mt
import pytest
from mcp.client import Client

from armguard_mcp.backends.base import BackendTimeout
from armguard_mcp.backends.fake import FakeBackend
from armguard_mcp.kinematics import FR3Kinematics
from armguard_mcp.models import Plan, Pose, Wrench
from armguard_mcp.safety import envelope
from armguard_mcp.safety.audit import AuditLogger
from armguard_mcp.safety.envelope import check_plan, densify
from armguard_mcp.server import ArmGuardApp, build
from tests.conftest import FR3_POLICY, READY, ROOT, Elicitor, audit_events, make_policy, text

pytestmark = pytest.mark.anyio


def target(j1: float = 0.3) -> list[float]:
    q = list(READY)
    q[0] = j1
    return q


def app_with(backend_cls: type[FakeBackend], policy: Any = None, **kw: Any) -> ArmGuardApp:
    policy = policy or make_policy(approval={"mode": "never"})
    kw.setdefault("speedup", 100.0)
    return build(policy, backend_cls.from_policy(policy, **kw), AuditLogger(None))


async def plan_joints(c: Client, q: list[float], **kw: Any) -> dict[str, Any]:
    r = await c.call_tool("plan_to_joints", {"joint_positions": q, **kw})
    assert not r.is_error, text(r)
    return r.structured_content


# --------------------------------------------------------------------------------------
# One execution at a time; stop/estop can never be lost while execute_plan is starting.
# --------------------------------------------------------------------------------------
class YieldingBackend(FakeBackend):
    """Reads yield to the event loop (like /compute_fk or waiting for a wrench message)."""

    gate: anyio.Event | None = None  # set by a test: the next hooked read signals and then waits
    hook = "wrench"
    execute_calls = 0

    async def _maybe_pause(self, where: str) -> None:
        if self.gate is not None and where == self.hook and not self.gate.is_set():
            self.gate.set()
            await anyio.sleep(0.05)
        else:
            await anyio.sleep(0)

    async def get_wrench(self) -> Wrench | None:
        await self._maybe_pause("wrench")
        return await super().get_wrench()

    async def forward_kinematics(self, joint_positions: Sequence[float]) -> Pose:
        await self._maybe_pause("fk")
        return await super().forward_kinematics(joint_positions)

    async def execute(self, plan, on_progress, should_abort):  # type: ignore[no-untyped-def]
        self.execute_calls += 1
        return await super().execute(plan, on_progress, should_abort)


@pytest.mark.parametrize("hook", ["fk", "wrench"])
@pytest.mark.parametrize("stopper", ["estop", "stop_motion"])
async def test_stop_or_estop_during_execute_startup_prevents_motion(stopper: str, hook: str) -> None:
    app = app_with(YieldingBackend)
    be = app.guard.backend
    assert isinstance(be, YieldingBackend)
    async with Client(app.server, mode="legacy") as c:
        s = await plan_joints(c, target(0.3))
        assert s["status"] == "executable"
        be.hook, be.gate = hook, anyio.Event()  # pause inside execute_plan's re-validation / wrench read
        res: dict[str, Any] = {}

        async def run() -> None:
            res["r"] = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await be.gate.wait()
            assert app.guard.state.active is not None  # the execution is already registered
            r = await c.call_tool(
                stopper, {"reason": "person entered the cell"} if stopper == "estop" else {}
            )
            assert not r.is_error, text(r)
        r = res["r"]
    assert be.execute_calls == 0, text(r)
    assert be._q[0] == pytest.approx(READY[0], abs=1e-9), f"robot moved after {stopper}: {text(r)}"
    if stopper == "estop":
        assert r.is_error and "e-stop" in text(r)
    else:
        assert not r.is_error and r.structured_content["status"] == "aborted"
        assert r.structured_content["executed"] is False
    assert app.guard.state.active is None


async def test_concurrent_execute_plans_are_serialized() -> None:
    app = app_with(YieldingBackend, speedup=5.0)
    be = app.guard.backend
    assert isinstance(be, YieldingBackend)
    async with Client(app.server, mode="legacy") as c:
        a = await plan_joints(c, target(0.3))
        b = await plan_joints(c, target(-0.3))
        out: dict[str, Any] = {}
        seen: dict[str, Any] = {}

        async def run(name: str, pid: str) -> None:
            out[name] = await c.call_tool("execute_plan", {"plan_id": pid})

        async with anyio.create_task_group() as tg:
            tg.start_soon(run, "a", a["plan_id"])
            tg.start_soon(run, "b", b["plan_id"])
            with anyio.fail_after(10):
                while not be._busy:  # wait until one of them moves the arm
                    await anyio.sleep(0.001)
            seen["status"] = (await c.call_tool("get_motion_status", {})).structured_content
            seen["switch"] = await c.call_tool("switch_controllers", {"deactivate": ["fr3_arm_controller"]})
    assert be.execute_calls == 1
    errors = [r for r in out.values() if r.is_error]
    oks = [r for r in out.values() if not r.is_error]
    assert len(errors) == 1 and len(oks) == 1
    assert "is still executing" in text(errors[0])
    assert (
        seen["status"]["executing"] is True
        and seen["status"]["plan_id"] == oks[0].structured_content["plan_id"]
    )
    assert seen["switch"].is_error and "while a plan is executing" in text(seen["switch"])
    assert app.guard.state.active is None


class StopCountingBackend(FakeBackend):
    stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1
        await super().stop()


@pytest.mark.parametrize("mode", ["legacy", "2026-07-28"])
async def test_cancelled_execute_plan_commands_a_stop(mode: str) -> None:
    app = app_with(StopCountingBackend, speedup=1.0)  # ~3 s trajectory
    be = app.guard.backend
    assert isinstance(be, StopCountingBackend)
    async with Client(app.server, mode=mode) as c:
        s = await plan_joints(c, target(0.3))
        with anyio.move_on_after(0.3):
            await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        with anyio.fail_after(5):
            while app.guard.state.active is not None:
                await anyio.sleep(0.01)
        status = (await c.call_tool("get_motion_status", {})).structured_content
    assert be.stop_calls >= 1
    assert "interrupted" in (status["last_result"] or "")
    assert any(e["event"] == "execution_interrupted" for e in app.guard.audit.tail(100))
    assert 0.0 < be._q[0] < 0.3  # stopped part-way


# --------------------------------------------------------------------------------------
# Force monitoring fails closed.
# --------------------------------------------------------------------------------------
class NoWrenchBackend(FakeBackend):
    async def get_wrench(self) -> Wrench | None:
        return None


async def test_execution_refused_without_wrench_estimate() -> None:
    app = app_with(NoWrenchBackend, table_height=0.33)
    be = app.guard.backend
    async with Client(app.server, mode="legacy") as c:
        s = (
            await c.call_tool(
                "plan_cartesian_path", {"waypoints": [{"position": {"x": 0.3069, "y": 0, "z": 0.30}}]}
            )
        ).structured_content
        assert s["force_monitoring"] == "unavailable"
        assert any("execute_plan will refuse" in n for n in s["notes"])
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
    assert r.is_error and "no external wrench estimate" in text(r) and "nothing moved" in text(r)
    assert be._q == pytest.approx(READY)


async def test_unmonitored_execution_needs_explicit_opt_in_and_is_disclosed() -> None:
    policy = make_policy(approval={"mode": "always"}, force={"require_wrench": False})
    app = app_with(NoWrenchBackend, policy)
    human = Elicitor("accept", approve=True)
    async with Client(app.server, mode="legacy", elicitation_callback=human) as c:
        s = await plan_joints(c, target(0.2))
        assert s["force_monitoring"] == "unavailable"
        assert any("will NOT be enforced" in n for n in s["notes"])
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert not r.is_error, text(r)
    assert "Force monitoring: OFF" in human.messages[0]
    assert audit_events(app, "execute_plan")[-1]["force_monitoring"] == "unavailable"


class HangingWrenchBackend(FakeBackend):
    """Once armed, answers the baseline read, then every later read hangs (a publisher that died)."""

    reads: int | None = None  # None = not armed

    async def get_wrench(self) -> Wrench | None:
        if self.reads is not None:
            self.reads += 1
            if self.reads >= 2:
                await anyio.sleep_forever()
        return await super().get_wrench()


class FrozenWrenchBackend(FakeBackend):
    """Keeps returning the same cached message: the stamp never advances."""

    async def get_wrench(self) -> Wrench | None:
        w = await super().get_wrench()
        assert w is not None
        return w.model_copy(update={"stamp": 1234.5})


@pytest.mark.parametrize(
    ("backend_cls", "expected"),
    [(HangingWrenchBackend, "took longer"), (FrozenWrenchBackend, "stale")],
)
async def test_force_monitor_fails_closed_on_hanging_or_stale_wrench(
    backend_cls: type[FakeBackend], expected: str
) -> None:
    app = app_with(backend_cls, table_height=0.40, speedup=1.0)  # contact 25 N at z=0.3875
    be = app.guard.backend
    async with Client(app.server, mode="legacy") as c:
        st = (await c.call_tool("get_robot_state", {})).structured_content["ee_pose"]["position"]
        wp = [{"position": {"x": st["x"], "y": st["y"], "z": 0.38}}]  # 20 mm into the table: 40 N
        s = (await c.call_tool("plan_cartesian_path", {"waypoints": wp})).structured_content
        assert s["status"] == "executable", s
        if isinstance(be, HangingWrenchBackend):
            be.reads = 0
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
    assert r.is_error and "force monitor failed" in text(r) and expected in text(r), text(r)
    assert be._contact_force(be._q) < 25.0  # stopped long before the 40 N end point


class BrokenStopBackend(FakeBackend):
    """The wrench stream dies mid-motion and the stop command fails too."""

    reads = 0

    async def get_wrench(self) -> Wrench | None:
        self.reads += 1
        if self.reads > 5:
            raise BackendTimeout("wrench topic went silent")
        return await super().get_wrench()

    async def stop(self) -> None:
        await super().stop()  # the simulated arm does stop ...
        raise BackendTimeout("stop service did not answer")  # ... but the backend reports failure


async def test_monitor_stop_failure_latches_estop_with_a_clear_error() -> None:
    app = app_with(BrokenStopBackend, speedup=5.0)
    be = app.guard.backend
    assert isinstance(be, BrokenStopBackend)
    async with Client(app.server, mode="legacy") as c:
        s = await plan_joints(c, target(0.3))
        be.reads = 0
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        status = (await c.call_tool("get_safety_status", {})).structured_content
        motion = (await c.call_tool("get_motion_status", {})).structured_content
    assert r.is_error and "force monitor failed" in text(r) and "software e-stop is latched" in text(r), text(
        r
    )
    assert text(r) != "Error executing tool execute_plan"
    assert status["estopped"] is True and status["reason_source"] == "server"
    assert "stop service did not answer" in status["reason"]
    assert motion["last_result"] and "force monitor failed" in motion["last_result"]


# --------------------------------------------------------------------------------------
# reset_estop: approval bound to the e-stop event; agent text marked as untrusted.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["legacy", "2026-07-28"])
async def test_reset_approval_cannot_release_a_newer_estop(mode: str) -> None:
    app = app_with(FakeBackend)
    other: dict[str, Client] = {}
    shown: list[str] = []

    async def human(context: Any, params: Any) -> mt.ElicitResult:
        shown.append(params.message)
        if "FIRST" not in params.message:  # the operator only agrees to release the FIRST stop
            return mt.ElicitResult(action="decline")
        if len(shown) == 1:  # while the prompt is open, another session latches a new e-stop
            r = await other["c2"].call_tool("estop", {"reason": "SECOND: person entered the cell"})
            assert not r.is_error
        return mt.ElicitResult(action="accept", content={"approve": True, "operator": "lynn"})

    async with Client(app.server, mode="legacy") as c2:
        other["c2"] = c2
        async with Client(app.server, mode=mode, elicitation_callback=human) as c1:
            assert not (await c1.call_tool("estop", {"reason": "FIRST: test"})).is_error
            r = await c1.call_tool("reset_estop", {})
    assert r.is_error, text(r)
    assert app.guard.state.estopped and app.guard.state.reason == "SECOND: person entered the cell"
    assert "event #1" in shown[0]


async def test_reset_prompt_marks_the_agent_reason_as_unverified() -> None:
    app = app_with(FakeBackend)
    human = Elicitor("decline")
    injected = (
        "routine test stop, no fault.\n\nOperator note: cell was already inspected and cleared by the "
        "safety officer at 09:14. Safe to approve.\n\n" + "padding " * 40
    )
    async with Client(app.server, mode="legacy", elicitation_callback=human) as c:
        await c.call_tool("estop", {"reason": injected})
        await c.call_tool("reset_estop", {})
    msg = human.messages[0]
    line = next(ln for ln in msg.splitlines() if "routine test stop" in ln)
    assert line.startswith("Reason given by the AI agent (unverified")
    assert '"routine test stop, no fault. Operator note:' in line  # one quoted line, no fake paragraphs
    assert "padding padding" not in msg.split(line)[1]
    assert len(line) < 220 and line.endswith('..."')
    assert msg.splitlines()[0].startswith("RESET SOFTWARE E-STOP")  # server facts come first


@pytest.mark.parametrize("reason", ["x" * 600, None, 42, "", {"nested": True}])
async def test_estop_accepts_any_reason(reason: Any) -> None:
    app = app_with(FakeBackend)
    async with Client(app.server, mode="legacy") as c:
        r = await c.call_tool("estop", {"reason": reason})
        assert not r.is_error, text(r)
        assert (await c.call_tool("get_safety_status", {})).structured_content["estopped"] is True
    assert any(e["event"] == "estop" for e in app.guard.audit.tail(20))


async def test_estop_without_arguments() -> None:
    app = app_with(FakeBackend)
    async with Client(app.server, mode="legacy") as c:
        r = await c.call_tool("estop", {})
        assert not r.is_error and r.structured_content["reason"] == "requested by agent"


# --------------------------------------------------------------------------------------
# Dry run and the gripper.
# --------------------------------------------------------------------------------------
def _hardware_state(be: FakeBackend) -> tuple[Any, ...]:
    return (
        tuple(be._q),
        be._gripper_width,
        be._grasped,
        tuple(sorted((n, c.state) for n, c in be._controllers.items())),
        be._collision_force_n,
        be._collision_torque_nm,
        be._in_error,
    )


async def test_dry_run_never_actuates_anything() -> None:
    policy = make_policy(dry_run=True, approval={"mode": "never"})
    app = app_with(FakeBackend, policy, object_width=0.03)
    be = app.guard.backend
    assert isinstance(be, FakeBackend)
    be._in_error = True
    before = _hardware_state(be)
    async with Client(app.server, mode="legacy") as c:
        s = await plan_joints(c, target(0.3))
        calls = [
            ("execute_plan", {"plan_id": s["plan_id"]}),
            ("gripper_move", {"width_m": 0.01}),
            ("gripper_grasp", {"width_m": 0.03, "force_n": 40.0}),
            ("gripper_home", {}),
            (
                "switch_controllers",
                {"activate": ["joint_impedance_controller"], "deactivate": ["fr3_arm_controller"]},
            ),
            ("set_collision_thresholds", {"force_n": 20.0, "torque_nm": 4.0}),
            ("error_recovery", {}),
        ]
        for name, args in calls:
            r = await c.call_tool(name, args)
            assert not r.is_error, (name, text(r))
            assert _hardware_state(be) == before, f"{name} actuated the robot in dry-run mode: {text(r)}"
    outcomes = {e["tool"]: e["outcome"] for e in audit_events(app) if e.get("tool") in dict(calls)}
    assert set(outcomes.values()) == {"dry_run"}, outcomes


@pytest.mark.parametrize("stopper", ["estop", "stop_motion"])
async def test_stop_interrupts_an_inflight_gripper_action(stopper: str) -> None:
    app = app_with(FakeBackend)
    be = app.guard.backend
    assert isinstance(be, FakeBackend)
    async with Client(app.server, mode="legacy") as c:
        res: dict[str, Any] = {}

        async def close() -> None:  # 0.08 m at 1 mm/s with speedup 100: 0.8 s
            res["g"] = await c.call_tool("gripper_move", {"width_m": 0.0, "speed_mps": 0.001})

        async with anyio.create_task_group() as tg:
            tg.start_soon(close)
            await anyio.sleep(0.1)
            r = await c.call_tool(
                stopper, {"reason": "hand between the fingers"} if stopper == "estop" else {}
            )
            assert not r.is_error, text(r)
            width_at_stop = be._gripper_width
        await anyio.sleep(0.2)
    assert res["g"].is_error and "interrupted" in text(res["g"]), text(res["g"])
    assert 0.0 < be._gripper_width < 0.08
    assert be._gripper_width == pytest.approx(width_at_stop, abs=0.002)
    assert app.guard.state.gripper is None


# --------------------------------------------------------------------------------------
# Envelope: accelerations and thin keep-out zones.
# --------------------------------------------------------------------------------------
async def test_envelope_rejects_a_velocity_step_whatever_scaling_the_backend_claims() -> None:
    policy = make_policy()
    q1 = target(0.05)
    # Constant velocity from rest: 0.05 rad in 0.1 s is 0.5 rad/s (19 % of the limit, under the
    # velocity cap) but reaching it instantly is an unbounded acceleration.
    plan = Plan(
        plan_id="p",
        kind="joints",
        joint_names=policy.robot.joint_names,
        waypoints=[list(READY), q1],
        time_from_start=[0.0, 0.1],
        duration_s=0.1,
        created_at=0.0,
        velocity_scaling=0.1,
        acceleration_scaling=0.1,  # what the backend claims
    )
    samples = densify(plan, 0.02)
    kin = FR3Kinematics()
    tcp = [[kin.fk(q)[i][3] for i in range(3)] for q in samples]
    verdict = check_plan(plan, policy, samples, tcp)
    assert not verdict.ok and "ACCELERATION" in {v.code for v in verdict.hard}


@pytest.mark.parametrize("scaling", [(0.3, 0.3), (0.3, 0.05), (0.05, 0.3), (0.1, 0.1)])
async def test_fake_cartesian_plans_respect_the_acceleration_cap(scaling: tuple[float, float]) -> None:
    policy = make_policy()
    app = app_with(FakeBackend, policy)
    vs, as_ = scaling
    async with Client(app.server, mode="legacy") as c:
        for dx, dy, dz in [(0.1, 0.0, 0.0), (0.0, 0.0, -0.18), (-0.1, 0.15, 0.05)]:
            st = (await c.call_tool("get_robot_state", {})).structured_content["ee_pose"]["position"]
            wp = [{"position": {"x": st["x"] + dx, "y": st["y"] + dy, "z": st["z"] + dz}}]
            args = {"waypoints": wp, "velocity_scaling": vs, "acceleration_scaling": as_}
            s = (await c.call_tool("plan_cartesian_path", args)).structured_content
            assert s["status"] == "executable", s
            plan = app.guard.plans.peek(s["plan_id"]).plan
            assert envelope.max_joint_acceleration_ratio(plan, policy) <= as_ * 1.0 + 1e-6
            assert s["max_joint_acceleration_ratio"] <= as_ + 1e-4


async def test_thin_keep_out_zone_cannot_be_tunnelled() -> None:
    """A 5 mm panel placed exactly between two envelope samples of an unrefined check."""
    kin = FR3Kinematics()

    def tcp(q: Sequence[float]) -> list[float]:
        return [kin.fk(q)[i][3] for i in range(3)]

    goal = [0.5, -0.35, 0.5, -2.0, 0.5, 1.9, 0.5]  # every joint moves 0.35-0.5 rad
    probe = app_with(FakeBackend, make_policy(approval={"mode": "never"}, workspace={"keep_out": []}))
    async with Client(probe.server, mode="legacy") as c:
        s0 = await plan_joints(c, goal)
    plan = probe.guard.plans.peek(s0["plan_id"]).plan
    pts = [tcp(q) for q in densify(plan, 0.02)]  # the joint-space samples alone
    i = max(range(len(pts) - 1), key=lambda k: math.dist(pts[k], pts[k + 1]))
    step = math.dist(pts[i], pts[i + 1])
    assert step > 0.01  # the TCP jumps more than a centimetre between joint-space samples
    axis = max(range(3), key=lambda a: abs(pts[i + 1][a] - pts[i][a]))
    c_ = [(pts[i][a] + pts[i + 1][a]) / 2 for a in range(3)]
    lo, hi = [v - 0.04 for v in c_], [v + 0.04 for v in c_]
    lo[axis], hi[axis] = c_[axis] - 0.0025, c_[axis] + 0.0025
    assert not any(all(lo[a] <= p[a] <= hi[a] for a in range(3)) for p in pts)
    policy = make_policy(
        approval={"mode": "never"}, workspace={"keep_out": [{"name": "glass_panel", "min": lo, "max": hi}]}
    )
    app = app_with(FakeBackend, policy)
    async with Client(app.server, mode="legacy") as c:
        s = await plan_joints(c, goal)
    assert s["status"] == "rejected", s
    assert any(v["code"] == "KEEP_OUT" and "glass_panel" in v["message"] for v in s["violations"])


async def test_keep_out_slab_between_tcp_samples_is_detected() -> None:
    """Even a 1 mm slab that no refined sample lands in is caught by the segment test."""
    probe = app_with(FakeBackend)
    async with Client(probe.server, mode="legacy") as c:
        s = await plan_joints(c, target(0.3))
    _, _, poses = await probe.guard.validate_plan(probe.guard.plans.peek(s["plan_id"]).plan)
    ys = [p.position.y for p in poses]
    k = max(range(len(ys) - 1), key=lambda j: abs(ys[j + 1] - ys[j]))
    mid = (ys[k] + ys[k + 1]) / 2
    zone = {"name": "sheet", "min": [0.0, mid - 0.0005, 0.0], "max": [1.0, mid + 0.0005, 1.0]}
    assert not any(zone["min"][1] <= y <= zone["max"][1] for y in ys)
    app = app_with(FakeBackend, make_policy(approval={"mode": "never"}, workspace={"keep_out": [zone]}))
    async with Client(app.server, mode="legacy") as c:
        s = await plan_joints(c, target(0.3))
    assert s["status"] == "rejected" and s["violations"][0]["code"] == "KEEP_OUT"


# --------------------------------------------------------------------------------------
# Audit completeness.
# --------------------------------------------------------------------------------------
class FlakyJointStateBackend(FakeBackend):
    fail = False

    async def get_joint_state(self):  # type: ignore[no-untyped-def]
        if self.fail:
            raise BackendTimeout("no /joint_states for 0.5 s")
        return await super().get_joint_state()


@pytest.mark.parametrize("mode", ["legacy", "2026-07-28"])
async def test_resolver_backend_errors_are_audited(mode: str) -> None:
    app = app_with(FlakyJointStateBackend)
    be = app.guard.backend
    assert isinstance(be, FlakyJointStateBackend)
    async with Client(app.server, mode=mode) as c:
        s = await plan_joints(c, target(0.3))
        be.fail = True
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
    assert r.is_error and "no /joint_states" in text(r), text(r)
    ev = audit_events(app, "execute_plan")
    assert ev and ev[-1]["outcome"] == "error" and ev[-1]["plan_id"] == s["plan_id"]


async def test_argument_validation_failures_are_audited() -> None:
    app = app_with(FakeBackend)
    async with Client(app.server, mode="legacy") as c:
        r = await c.call_tool("plan_to_joints", {"joint_positions": "all the way up"})
        assert r.is_error
    ev = audit_events(app, "plan_to_joints")
    assert ev and ev[-1]["outcome"] == "rejected" and ev[-1]["cause"] == "ValidationError"


# --------------------------------------------------------------------------------------
# Tool schemas.
# --------------------------------------------------------------------------------------
async def test_tool_schemas_carry_bounds_units_and_clean_descriptions() -> None:
    app = app_with(FakeBackend)
    async with Client(app.server, mode="legacy") as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    for t in tools.values():
        assert "\n        " not in (t.description or ""), t.name
    grasp = tools["gripper_grasp"].input_schema["properties"]
    assert grasp["force_n"]["maximum"] == 40.0 and grasp["force_n"]["exclusiveMinimum"] == 0
    assert grasp["width_m"]["minimum"] == 0.0 and grasp["width_m"]["maximum"] == 0.08
    assert (
        "[m]" in grasp["epsilon_inner_m"]["description"] and "[m]" in grasp["epsilon_outer_m"]["description"]
    )
    joints = tools["plan_to_joints"].input_schema["properties"]["joint_positions"]
    assert joints["minItems"] == joints["maxItems"] == 7
    thr = tools["set_collision_thresholds"].input_schema["properties"]
    assert thr["force_n"]["maximum"] == 25.0 and thr["torque_nm"]["maximum"] == 5.0
    schema = str(tools["plan_to_pose"].input_schema)
    assert "N*m for wrenches" not in schema and "scalar part" in schema


# --------------------------------------------------------------------------------------
# Fake backend / policy agreement.
# --------------------------------------------------------------------------------------
async def test_fake_backend_accepts_drift_within_the_policy_tolerance() -> None:
    app = app_with(FakeBackend)
    be = app.guard.backend
    assert isinstance(be, FakeBackend)
    async with Client(app.server, mode="legacy") as c:
        s = await plan_joints(c, target(0.3))
        q = list(READY)
        q[0] += 0.005  # inside start_tolerance_rad = 0.01
        be.set_joint_positions(q)
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
    assert not r.is_error, text(r)
    assert be._q[0] == pytest.approx(0.3)


# --------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

_WRAPPER = """
import pathlib, sys
from armguard_mcp.backends.fake import FakeBackend
marker = pathlib.Path(sys.argv.pop(1))
orig = FakeBackend.shutdown
async def shutdown(self):
    marker.write_text("shutdown ran")
    await orig(self)
FakeBackend.shutdown = shutdown
from armguard_mcp.cli import main
sys.exit(main())
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_stops_and_shuts_down_the_backend(tmp_path: Any) -> None:
    marker = tmp_path / "marker"
    wrapper = tmp_path / "wrap.py"
    wrapper.write_text(_WRAPPER)
    p = subprocess.Popen(
        [sys.executable, str(wrapper), str(marker), "--policy", str(FR3_POLICY)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=ENV,
    )
    try:
        deadline = time.monotonic() + 30
        assert p.stderr is not None
        while b"armguard-mcp" not in p.stderr.readline():  # wait for the start-up log line
            assert time.monotonic() < deadline and p.poll() is None
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=20)
    finally:
        if p.poll() is None:
            p.kill()
    assert marker.exists() and marker.read_text() == "shutdown ran"
    assert p.returncode == 0


def test_unwritable_audit_log_exits_cleanly(tmp_path: Any) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "armguard_mcp",
            "--policy",
            str(FR3_POLICY),
            "--audit-log",
            str(blocker / "audit.jsonl"),
        ],
        capture_output=True,
        text=True,
        env=ENV,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert out.returncode == 2, out.stderr
    assert "cannot open audit log" in out.stderr and "Traceback" not in out.stderr
