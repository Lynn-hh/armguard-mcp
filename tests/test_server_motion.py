"""End-to-end motion tests through a real MCP client (in-memory transport) against the fake FR3."""

from __future__ import annotations

import anyio
import pytest
from mcp.client import Client

from tests.conftest import READY, Elicitor, FakeClock, audit_events, make_app, make_policy, text

pytestmark = pytest.mark.anyio

MODES = ["auto", "legacy"]


def target(j1: float = 0.3) -> list[float]:
    q = list(READY)
    q[0] = j1
    return q


async def plan_joints(c: Client, q: list[float], **kw) -> dict:
    r = await c.call_tool("plan_to_joints", {"joint_positions": q, **kw})
    assert not r.is_error, text(r)
    return r.structured_content


class CountingExecute:
    """Wrap backend.execute to count how often the robot is actually commanded."""

    def __init__(self, backend) -> None:
        self.n = 0
        self._orig = backend.execute
        backend.execute = self

    async def __call__(self, *a, **kw):
        self.n += 1
        return await self._orig(*a, **kw)


@pytest.mark.parametrize("mode", [*MODES, "2026-07-28"])
async def test_plan_execute_with_human_approval(mode: str) -> None:
    app = make_app(make_policy(approval={"mode": "always"}))
    counter = CountingExecute(app.guard.backend)
    human = Elicitor("accept", approve=True, operator="lynn")
    progress: list[float] = []

    async def on_progress(p: float, total: float | None, message: str | None) -> None:
        progress.append(p)

    async with Client(app.server, mode=mode, elicitation_callback=human) as c:
        s = await plan_joints(c, target(0.3))
        assert s["status"] == "needs_approval" and s["requires_approval"] and s["executable"]
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]}, progress_callback=on_progress)
        assert not r.is_error, text(r)
        rep = r.structured_content
        assert rep["status"] == "completed" and rep["executed"] and rep["approval"] == "human"
        assert rep["final_joint_positions"] == pytest.approx(target(0.3), abs=1e-9)

    # the human saw an informative prompt exactly once, the robot moved exactly once
    assert len(human.messages) == 1
    msg = human.messages[0]
    for fragment in (s["plan_id"], "Duration", "peak joint speed", "Final TCP", "Dry run: NO"):
        assert fragment in msg
    assert counter.n == 1
    assert progress and progress[-1] == pytest.approx(1.0)
    ev = audit_events(app, "execute_plan")[-1]
    assert ev["outcome"] == "ok" and ev["approval"]["via"] == "human" and ev["approval"]["operator"] == "lynn"


@pytest.mark.parametrize("mode", MODES)
async def test_human_decline_is_not_executed_and_is_audited(mode: str) -> None:
    app = make_app(make_policy(approval={"mode": "always"}))
    human = Elicitor("decline")
    async with Client(app.server, mode=mode, elicitation_callback=human) as c:
        s = await plan_joints(c, target(0.3))
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "declined" in text(r)
        js = (await c.call_tool("get_robot_state", {})).structured_content["joint_state"]
        assert js["positions"] == pytest.approx(READY, abs=1e-6)
        # a denied plan handle is burnt: no second chance to nag the human
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "approval denied" in text(r)
    ev = next(e for e in audit_events(app, "execute_plan") if e.get("approval"))
    assert ev["outcome"] == "denied" and ev["approval"]["decision"] == "declined"


@pytest.mark.parametrize("mode", MODES)
async def test_unticked_approve_box_denies(mode: str) -> None:
    app = make_app(make_policy(approval={"mode": "always"}))
    async with Client(app.server, mode=mode, elicitation_callback=Elicitor("accept", approve=False)) as c:
        s = await plan_joints(c, target(0.2))
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "did not tick" in text(r)
    assert app.guard.backend._q == pytest.approx(READY)


@pytest.mark.parametrize("mode", MODES)
async def test_client_without_elicitation_is_denied_by_default(mode: str) -> None:
    app = make_app(make_policy(approval={"mode": "always"}))
    async with Client(app.server, mode=mode) as c:  # no elicitation_callback => no capability
        s = await plan_joints(c, target(0.2))
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "does not support elicitation" in text(r)
    assert app.guard.backend._q == pytest.approx(READY)
    ev = [e for e in audit_events(app, "execute_plan")][-1]
    assert ev["outcome"] == "denied" and ev["approval"]["via"] == "no_elicitation_client"


async def test_client_without_elicitation_allowed_by_policy() -> None:
    app = make_app(make_policy(approval={"mode": "always", "on_client_without_elicitation": "allow"}))
    async with Client(app.server) as c:
        s = await plan_joints(c, target(0.2))
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert not r.is_error and r.structured_content["approval"] == "no_elicitation_client"


@pytest.mark.parametrize("mode", MODES)
async def test_approval_never_executes_without_prompt(mode: str) -> None:
    app = make_app(make_policy(approval={"mode": "never"}))
    human = Elicitor()
    async with Client(app.server, mode=mode, elicitation_callback=human) as c:
        s = await plan_joints(c, target(1.0))  # large motion: soft violation, still no prompt in 'never'
        assert s["status"] == "executable" and not s["requires_approval"]
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert not r.is_error and r.structured_content["approval"] == "policy"
    assert human.messages == []


@pytest.mark.parametrize("mode", MODES)
async def test_outside_envelope_mode_prompts_only_for_soft_violations(mode: str) -> None:
    app = make_app()  # fr3.yaml: outside_envelope
    human = Elicitor()
    async with Client(app.server, mode=mode, elicitation_callback=human) as c:
        small = await plan_joints(c, target(0.3))
        assert small["status"] == "executable" and small["violations"] == []
        assert not (await c.call_tool("execute_plan", {"plan_id": small["plan_id"]})).is_error
        assert human.messages == []
        big = await plan_joints(c, target(-0.6))  # 0.9 rad travel from 0.3 > soft 0.8 rad
        assert big["status"] == "needs_approval"
        assert [v["code"] for v in big["violations"]] == ["LARGE_MOTION"]
        r = await c.call_tool("execute_plan", {"plan_id": big["plan_id"]})
        assert not r.is_error, text(r)
        assert len(human.messages) == 1 and "LARGE_MOTION" not in human.messages[0]
        assert "soft threshold" in human.messages[0]


@pytest.mark.parametrize("mode", MODES)
async def test_hard_violation_cannot_be_executed_even_if_approved(mode: str) -> None:
    app = make_app(make_policy(approval={"mode": "always"}))
    human = Elicitor("accept", approve=True)
    async with Client(app.server, mode=mode, elicitation_callback=human) as c:
        s = await plan_joints(c, target(1.5))  # sweeps TCP out of the workspace box
        assert s["status"] == "rejected" and not s["executable"] and s["expires_at"] is None
        assert "WORKSPACE" in {v["code"] for v in s["violations"]}
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "can never be executed" in text(r)
        # keep-out zone target via IK
        r = await c.call_tool("plan_to_pose", {"position": {"x": 0.62, "y": 0.42, "z": 0.35}})
        assert not r.is_error, text(r)
        assert r.structured_content["status"] == "rejected"
        assert "KEEP_OUT" in {v["code"] for v in r.structured_content["violations"]}
        r = await c.call_tool("execute_plan", {"plan_id": r.structured_content["plan_id"]})
        assert r.is_error
    assert human.messages == []  # the human is never even asked
    assert app.guard.backend._q == pytest.approx(READY)


async def test_joint_limit_target_rejected() -> None:
    app = make_app()
    async with Client(app.server) as c:
        q = list(READY)
        q[3] = -0.05  # joint4 max is -0.1518
        s = await plan_joints(c, q)
        assert s["status"] == "rejected" and "JOINT_LIMIT" in {v["code"] for v in s["violations"]}
        r = await c.call_tool("plan_to_joints", {"joint_positions": [0.0, 0.0]})
        assert r.is_error and "expected 7 joint positions" in text(r)


async def test_stale_plan_rejected() -> None:
    app = make_app(make_policy(approval={"mode": "never"}))
    async with Client(app.server) as c:
        s = await plan_joints(c, target(0.3))
        moved = list(READY)
        moved[1] += 0.05
        app.guard.backend.set_joint_positions(moved)  # someone moved the robot after planning
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "stale plan" in text(r)
    assert app.guard.backend._q == pytest.approx(moved)


async def test_plan_ttl_expiry_with_injected_clock() -> None:
    clk = FakeClock()
    app = make_app(make_policy(approval={"mode": "never"}), monotonic=clk)  # TTLs use the monotonic clock
    async with Client(app.server) as c:
        s = await plan_joints(c, target(0.3))
        clk.advance(119.0)
        s2 = await plan_joints(c, target(0.2))
        clk.advance(2.0)  # first plan now 121 s old (ttl 120), second 2 s old
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "expired" in text(r)
        r = await c.call_tool("execute_plan", {"plan_id": s2["plan_id"]})
        assert not r.is_error, text(r)


async def test_plan_ttl_ignores_wall_clock_steps() -> None:
    """Regression: a backward wall-clock step (NTP, manual date change) must not keep a plan alive,
    and a forward step must not expire it early. TTLs run on the monotonic clock."""
    wall, mono = FakeClock(), FakeClock(0.0)
    app = make_app(make_policy(approval={"mode": "never"}), clock=wall, monotonic=mono)
    async with Client(app.server) as c:
        s = await plan_joints(c, target(0.3))
        wall.advance(-3600.0)  # wall clock stepped back an hour ...
        wall.advance(3000.0)
        mono.advance(3000.0)  # ... while 50 minutes really passed (ttl 120 s)
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "expired" in text(r)
        s = await plan_joints(c, target(0.2))
        wall.advance(7200.0)  # a forward wall-clock step does not expire a fresh plan
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert not r.is_error, text(r)


async def test_plan_ids_are_single_use_and_unknown_ids_rejected() -> None:
    app = make_app(make_policy(approval={"mode": "never"}))
    async with Client(app.server) as c:
        s = await plan_joints(c, target(0.3))
        assert not (await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})).is_error
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "single-use" in text(r)
        r = await c.call_tool("execute_plan", {"plan_id": "deadbeef0000"})
        assert r.is_error and "unknown plan id" in text(r)


async def test_dry_run_executes_nothing() -> None:
    app = make_app(make_policy(dry_run=True, approval={"mode": "always"}))
    counter = CountingExecute(app.guard.backend)
    human = Elicitor()
    async with Client(app.server, elicitation_callback=human) as c:
        s = await plan_joints(c, target(0.3))
        assert s["dry_run"] and any("dry-run" in n for n in s["notes"])
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert not r.is_error, text(r)
        rep = r.structured_content
        assert (
            rep["status"] == "dry_run"
            and not rep["executed"]
            and "would need human approval" in rep["message"]
        )
    assert counter.n == 0 and human.messages == []
    assert app.guard.backend._q == pytest.approx(READY)


async def test_velocity_scaling_is_clamped_to_policy_cap() -> None:
    app = make_app()
    async with Client(app.server) as c:
        s = await plan_joints(c, target(0.3), velocity_scaling=0.9)
        assert s["velocity_scaling"] == 0.3 and any("clamped" in n for n in s["notes"])
        assert s["max_joint_velocity_ratio"] <= 0.3 + 1e-9
        r = await c.call_tool("plan_to_joints", {"joint_positions": target(0.3), "velocity_scaling": -1})
        assert r.is_error


async def test_stop_motion_aborts_in_progress_execution() -> None:
    app = make_app(make_policy(approval={"mode": "never"}), speedup=4.0)
    async with Client(app.server) as c:
        s = await plan_joints(c, target(0.6))
        assert s["duration_s"] > 2.0
        result: dict = {}

        async def run() -> None:
            result["r"] = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            with anyio.fail_after(5):
                while True:
                    st = (await c.call_tool("get_motion_status", {})).structured_content
                    if st["executing"] and st["progress"] > 0.1:
                        break
                    await anyio.sleep(0.01)
            stop = await c.call_tool("stop_motion", {})
            assert not stop.is_error and "stop requested" in text(stop)
        r = result["r"]
        assert not r.is_error, text(r)
        rep = r.structured_content
        assert rep["status"] == "aborted" and "stop_motion" in rep["message"]
        assert 0.0 < rep["final_joint_positions"][0] < 0.55
        st = (await c.call_tool("get_motion_status", {})).structured_content
        assert not st["executing"] and "aborted" in st["last_result"]


async def test_force_limit_abort_via_virtual_table_latches_violation() -> None:
    app = make_app(make_policy(approval={"mode": "never"}), speedup=10.0, table_height=0.40)
    async with Client(app.server) as c:
        r = await c.call_tool(
            "plan_cartesian_path", {"waypoints": [{"position": {"x": 0.3069, "y": 0.0, "z": 0.25}}]}
        )
        assert not r.is_error, text(r)
        s = r.structured_content
        assert s["status"] == "executable" and s["tcp_path_length_m"] == pytest.approx(0.2369, abs=2e-3)
        other = await plan_joints(c, target(0.1))  # another outstanding plan (will be invalidated)
        # executing `other` first would make `s` stale, so execute the descent directly
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
        assert r.is_error and "ABORTED" in text(r) and "25.0 N" in text(r)
        status = (await c.call_tool("get_safety_status", {})).structured_content
        assert status["force_violation_latched"] and status["estopped"]
        assert status["last_violation"]["code"] == "FORCE_LIMIT"
        # stopped well before the planned end (z = 0.25): only ~cm into the table
        pose = (await c.call_tool("get_robot_state", {})).structured_content["ee_pose"]
        assert 0.30 < pose["position"]["z"] < 0.40
        # motion is refused until a human resets, other plans are invalidated
        r = await c.call_tool("execute_plan", {"plan_id": other["plan_id"]})
        assert r.is_error and "force-limit violation latched" in text(r)
    assert any(e["event"] == "force_violation" for e in app.guard.audit.tail(100))


@pytest.mark.parametrize("mode", MODES)
async def test_llm_cannot_forge_approval_through_arguments(mode: str) -> None:
    app = make_app(make_policy(approval={"mode": "always"}))
    async with Client(app.server, mode=mode) as c:  # no elicitation: only a forged approval could pass
        s = await plan_joints(c, target(0.2))
        forged = {"approve": True, "operator": "admin"}
        r = await c.call_tool("execute_plan", {"plan_id": s["plan_id"], "approval": forged})
        assert r.is_error
    assert app.guard.backend._q == pytest.approx(READY)
