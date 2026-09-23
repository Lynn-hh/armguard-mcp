"""The armguard MCP server: tools for an LLM, with a safety envelope enforced outside the LLM.

Every tool call follows the same path:

    rate limit -> e-stop check (actuating tools) -> validation -> approval -> action -> audit

Approval uses MCP elicitation through a ``Resolve``-injected parameter. Resolvers are pure
(under the 2026-07-28 protocol they can run more than once per call); all side effects -
motion, plan consumption, audit of the decision - happen in the tool body, which re-checks
everything the resolver looked at.

NOTE: this file intentionally does not use ``from __future__ import annotations``: the MCP
SDK resolves tool/resolver type hints at registration time, and the ``Resolve(...)`` markers
reference closures that only exist inside :func:`build`.
"""

import base64
import json
import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NoReturn

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resolve import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
    Elicit,
    ElicitationResult,
    Resolve,
)
from mcp.server.request_state import RequestStateSecurity
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from armguard_mcp import __version__
from armguard_mcp import geometry as g
from armguard_mcp.backends.base import BackendError, RobotBackend
from armguard_mcp.imaging import downscale, pillow_available
from armguard_mcp.models import (
    ActionResult,
    ControllerList,
    EnvelopeVerdict,
    ExecutionReport,
    GraphInfo,
    MotionStatus,
    Plan,
    PlanSummary,
    Pose,
    Quaternion,
    RobotState,
    SafetyStatus,
    Vector3,
    Violation,
    Wrench,
)
from armguard_mcp.plans import PlanError, PlanStore, StoredPlan, max_deviation
from armguard_mcp.policy import (
    TOOL_GROUPS,
    ApprovalSection,
    ForceSection,
    GripperSection,
    JointLimit,
    MotionSection,
    Policy,
    RateLimitsSection,
    WorkspaceSection,
)
from armguard_mcp.safety.audit import AuditLogger, iso_utc
from armguard_mcp.safety.envelope import (
    check_plan,
    check_wrench,
    clamp_scaling,
    densify,
    joint_travel,
    max_joint_velocity_ratio,
    path_length,
)
from armguard_mcp.safety.ratelimit import RateLimiter, RateLimitExceeded
from armguard_mcp.safety.state import ActiveExecution, SafetyState

logger = logging.getLogger(__name__)

_RETREAT_HYSTERESIS_N = 1.0
_RETREAT_HYSTERESIS_NM = 0.2

SERVER_NAME = "armguard-mcp"

INSTRUCTIONS = """\
armguard-mcp controls a real robot arm through a server-side safety envelope.
Workflow: call get_safety_envelope first and plan within it. Motion is two-step: plan_to_joints /
plan_to_pose / plan_cartesian_path return a plan summary and a plan_id; inspect it, then call
execute_plan(plan_id). Plans are single-use and expire. A human may be asked to approve execution;
never try to work around a denial. Limits (joint limits, workspace box, keep-out zones, speed caps,
step sizes, force thresholds, rate limits) are enforced by the server, not by you: rejected plans can
never be executed. stop_motion and estop are always available and never rate limited; use them
whenever anything looks wrong. Units: metres, radians, seconds, newtons.
"""


# --------------------------------------------------------------------------------------
# Approval models
# --------------------------------------------------------------------------------------
class ApprovalForm(BaseModel):
    """The form shown to the human operator (MCP elicitation)."""

    approve: bool = Field(
        default=False, description="Tick to APPROVE this robot action. Leave unticked to deny."
    )
    operator: str = Field(default="", description="Your name or initials (recorded in the audit log)")


class PolicyDecision(BaseModel):
    """An approval outcome decided by policy, without asking a human."""

    approved: bool
    via: Literal["not_required", "no_elicitation_client", "precheck_failed"]
    reason: str


ApprovalOutcome = ElicitationResult[ApprovalForm]


class Denied(ToolError):
    """A refusal by the safety layer (audited as ``denied`` rather than ``error``)."""


# --------------------------------------------------------------------------------------
# Tool I/O models
# --------------------------------------------------------------------------------------
class CartesianWaypoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: Vector3 = Field(description="TCP position [m]")
    orientation: Quaternion | None = Field(
        default=None, description="TCP orientation (x, y, z, w); omit to keep the current orientation"
    )


class SafetyEnvelopeInfo(BaseModel):
    summary: str
    robot: str
    base_frame: str
    ee_frame: str
    joint_names: list[str]
    joint_limits: dict[str, JointLimit]
    home_joint_positions: list[float]
    workspace: WorkspaceSection
    motion: MotionSection
    force: ForceSection
    gripper: GripperSection | None
    controllers_allowlist: list[str]
    camera_topics: list[str]
    max_image_width: int
    approval: ApprovalSection
    rate_limits: RateLimitsSection
    dry_run: bool
    enabled_tool_groups: list[str]


class AuditTail(BaseModel):
    events: list[dict[str, Any]]


# --------------------------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------------------------
@dataclass
class _CallRecord:
    tool: str
    args: dict[str, Any]
    plan_id: str | None = None
    verdict: Any = None
    approval: dict[str, Any] | None = None
    outcome: str = "ok"
    extra: dict[str, Any] = field(default_factory=dict)


class ArmGuard:
    """Policy enforcement + orchestration, independent of the MCP wiring."""

    def __init__(
        self,
        policy: Policy,
        backend: RobotBackend,
        audit: AuditLogger | None = None,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.backend = backend
        self.audit = audit if audit is not None else AuditLogger(clock=clock)
        self.clock = clock
        self.state = SafetyState(dry_run=policy.dry_run, clock=clock)
        self.plans = PlanStore(policy.motion.plan_ttl_s, clock=clock)
        self.rate = RateLimiter(policy.rate_limits, clock=monotonic)
        self._started = False
        self._start_lock = anyio.Lock()

    # --- plumbing ----------------------------------------------------------------------
    async def ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if not self._started:
                await self.backend.start()
                self._started = True

    @asynccontextmanager
    async def call(self, tool: str, args: dict[str, Any] | None = None) -> AsyncIterator[_CallRecord]:
        """Rate limit + audit wrapper around every tool body."""
        rec = _CallRecord(tool=tool, args=dict(args or {}))
        try:
            try:
                self.rate.acquire(tool)
            except RateLimitExceeded as e:
                raise Denied(str(e)) from e
            await self.ensure_started()
            yield rec
        except Denied as e:
            self._audit_call(rec, "denied", error=str(e))
            raise
        except ToolError as e:
            self._audit_call(rec, "error", error=str(e))
            raise
        except BackendError as e:
            self._audit_call(rec, "error", error=f"{type(e).__name__}: {e}")
            raise ToolError(f"robot backend error: {e}") from e
        except PlanError as e:
            self._audit_call(rec, "denied", error=str(e))
            raise Denied(str(e)) from e
        except BaseException as e:
            if isinstance(e, Exception):
                self._audit_call(rec, "error", error=f"{type(e).__name__}: {e}")
            else:  # cancellation: still leave a trace
                self._audit_call(rec, "cancelled", error=type(e).__name__)
            raise
        else:
            self._audit_call(rec, rec.outcome)

    def _audit_call(self, rec: _CallRecord, outcome: str, error: str | None = None) -> None:
        self.audit.log(
            "tool_call",
            tool=rec.tool,
            args=rec.args,
            outcome=outcome,
            plan_id=rec.plan_id,
            verdict=rec.verdict,
            approval=rec.approval,
            error=error,
            **rec.extra,
        )

    @staticmethod
    def deny(message: str) -> NoReturn:
        raise Denied(message)

    def require_motion_allowed(self) -> None:
        reason = self.state.motion_blocked_reason
        if reason is not None:
            self.deny(f"refused: {reason}")

    # --- approval ----------------------------------------------------------------------
    def approval_required(self, klass: str, inside_envelope: bool = False) -> bool:
        ap = self.policy.approval
        if klass == "reset_estop":
            return True
        if klass not in ap.require_for or ap.mode == "never":
            return False
        if ap.mode == "always":
            return True
        # outside_envelope: motion inside the envelope runs without a prompt; state-changing
        # operations that have no envelope notion always count as outside it.
        return not inside_envelope if klass == "execute" else True

    def approval_request(
        self, ctx: Context, *, required: bool, message: str, precheck_error: str | None = None
    ) -> "PolicyDecision | Elicit[ApprovalForm]":
        """Pure resolver core: decide whether to ask the human, never mutate anything."""
        if precheck_error is not None:
            return PolicyDecision(approved=False, via="precheck_failed", reason=precheck_error)
        if not required:
            return PolicyDecision(approved=True, via="not_required", reason="approval not required by policy")
        caps = ctx.client_capabilities
        elicitation = caps.elicitation if caps is not None else None
        can_elicit = elicitation is not None and (elicitation.form is not None or elicitation.url is None)
        if not can_elicit:
            allow = self.policy.approval.on_client_without_elicitation == "allow"
            return PolicyDecision(
                approved=allow,
                via="no_elicitation_client",
                reason=(
                    "client cannot ask a human (no elicitation support); policy allows proceeding"
                    if allow
                    else "human approval is required but this MCP client does not support elicitation; "
                    "policy approval.on_client_without_elicitation is 'deny'"
                ),
            )
        return Elicit(message, ApprovalForm)

    def check_approval(self, outcome: Any, required: bool, rec: _CallRecord) -> None:
        """Authoritative, body-side interpretation of the resolver outcome. Raises Denied."""
        decision: dict[str, Any]
        if isinstance(outcome, AcceptedElicitation):
            data = outcome.data
            if isinstance(data, ApprovalForm):
                decision = {
                    "decision": "approved" if data.approve else "denied",
                    "via": "human",
                    "operator": data.operator or None,
                    "required": required,
                }
                rec.approval = decision
                if not data.approve:
                    self.deny("denied: the human operator did not tick 'approve'")
                return
            if isinstance(data, PolicyDecision):
                if data.via == "not_required" and required:
                    decision = {
                        "decision": "denied",
                        "via": "policy",
                        "reason": "state changed during approval",
                    }
                    rec.approval = decision
                    self.deny("denied: approval became required while the request was in flight; retry")
                if not required and data.via == "not_required":
                    rec.approval = {"decision": "not_required", "via": "policy"}
                    return
                rec.approval = {"decision": "approved" if data.approved else "denied", "via": data.via}
                if not data.approved:
                    self.deny(f"denied: {data.reason}")
                return
            rec.approval = {"decision": "denied", "via": "unknown"}
            self.deny("denied: unrecognised approval outcome")
        if isinstance(outcome, DeclinedElicitation):
            rec.approval = {"decision": "declined", "via": "human", "required": required}
            self.deny("denied: the human operator declined")
        if isinstance(outcome, CancelledElicitation):
            rec.approval = {"decision": "cancelled", "via": "human", "required": required}
            self.deny("denied: the human operator cancelled the approval request")
        rec.approval = {"decision": "denied", "via": "unknown"}
        self.deny("denied: missing approval outcome")

    # --- planning ----------------------------------------------------------------------
    async def validate_plan(self, plan: Plan) -> tuple[EnvelopeVerdict, list[list[float]], list[Pose]]:
        samples = densify(plan, self.policy.motion.check_resolution_rad) if len(plan.waypoints) >= 1 else []
        poses = [await self.backend.forward_kinematics(q) for q in samples]
        verdict = check_plan(plan, self.policy, samples, [p.position.as_tuple() for p in poses])
        return verdict, samples, poses

    async def finalize_plan(self, plan: Plan, notes: list[str]) -> PlanSummary:
        verdict, _samples, poses = await self.validate_plan(plan)
        final_pose = await self.backend.forward_kinematics(plan.final)
        requires = self.approval_required("execute", verdict.inside_envelope) and verdict.ok
        status: Literal["executable", "needs_approval", "rejected"] = (
            "rejected" if not verdict.ok else ("needs_approval" if requires else "executable")
        )
        if self.state.dry_run:
            notes = [*notes, "server is in dry-run mode: execute_plan will validate only, nothing will move"]
        if not verdict.ok:
            notes = [
                *notes,
                "this plan violates hard limits and can never be executed; plan a different motion",
            ]
        expires = self.plans.next_expiry()
        travel = joint_travel(plan)
        summary = PlanSummary(
            plan_id=plan.plan_id,
            kind=plan.kind,
            status=status,
            executable=verdict.ok,
            duration_s=round(plan.duration_s, 4),
            num_waypoints=len(plan.waypoints),
            start_joint_positions=[round(v, 5) for v in plan.start],
            final_joint_positions=[round(v, 5) for v in plan.final],
            final_ee_pose=final_pose,
            max_joint_velocity_ratio=round(max_joint_velocity_ratio(plan, self.policy), 4),
            max_joint_travel_rad=round(max(travel) if travel else 0.0, 4),
            tcp_path_length_m=round(path_length([p.position.as_tuple() for p in poses]), 4),
            velocity_scaling=plan.velocity_scaling,
            acceleration_scaling=plan.acceleration_scaling,
            violations=verdict.violations,
            requires_approval=requires,
            expires_at=iso_utc(expires) if verdict.ok else None,
            dry_run=self.state.dry_run,
            notes=notes,
        )
        self.plans.put(plan, summary, verdict, expires_at=expires)
        for v in verdict.hard:
            self.state.record_violation(v)
        return summary

    def scaling(self, vel: float | None, acc: float | None) -> tuple[float, float, list[str]]:
        m = self.policy.motion
        try:
            vs, n1 = clamp_scaling(vel, m.default_velocity_scaling, m.max_velocity_scaling)
            as_, n2 = clamp_scaling(acc, m.default_acceleration_scaling, m.max_acceleration_scaling)
        except ValueError as e:
            self.deny(str(e))
        return vs, as_, [n for n in (n1, n2) if n]

    async def to_base_frame(
        self, position: Vector3, orientation: Quaternion | None, frame_id: str | None
    ) -> Pose:
        """Express a requested TCP pose in the robot base frame (orientation None = keep current)."""
        base = self.policy.robot.base_frame
        in_base = not frame_id or frame_id == base
        if orientation is None:
            current = await self.backend.get_ee_pose()
            q = current.orientation.as_tuple()
            if not in_base:
                # current orientation re-expressed in the requested frame
                tf = await self.backend.lookup_transform(frame_id, base)  # type: ignore[arg-type]
                r = g.matmul(g.matrix_from_quat(tf.orientation.as_tuple()), g.matrix_from_quat(q))
                q = g.quat_from_matrix(r)
        else:
            q = orientation.as_tuple()
            norm = math.sqrt(sum(c * c for c in q))
            if not math.isfinite(norm) or abs(norm - 1.0) > 0.05:
                self.deny(f"orientation quaternion must be unit length (got norm {norm:.3f})")
            q = g.quat_normalize(q)
        if not all(math.isfinite(c) for c in position.as_tuple()):
            self.deny("position must be finite")
        if in_base:
            return Pose(frame_id=base, position=position, orientation=Quaternion.of(q))
        tf = await self.backend.lookup_transform(base, frame_id)  # type: ignore[arg-type]
        t = g.matmul(
            g.matrix_from_quat(tf.orientation.as_tuple(), tf.position.as_tuple()),
            g.matrix_from_quat(q, position.as_tuple()),
        )
        return Pose(
            frame_id=base,
            position=Vector3.of(g.translation(t)),
            orientation=Quaternion.of(g.quat_from_matrix(t)),
        )

    def execute_message(self, stored: StoredPlan) -> str:
        s, pol = stored.summary, self.policy
        p = s.final_ee_pose.position
        warnings = "; ".join(v.message for v in s.violations) or "none"
        return (
            f"APPROVE ROBOT MOTION on '{pol.robot.name}'? Plan {s.plan_id} ({s.kind}: {stored.plan.source_request}).\n"
            f"Duration {s.duration_s:.2f} s, {s.num_waypoints} waypoints, peak joint speed "
            f"{100 * s.max_joint_velocity_ratio:.0f}% of limit (cap {100 * pol.motion.max_velocity_scaling:.0f}%), "
            f"largest joint travel {s.max_joint_travel_rad:.3f} rad, TCP path {s.tcp_path_length_m:.3f} m.\n"
            f"Final TCP in {pol.robot.base_frame}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} m.\n"
            f"Soft warnings: {warnings}.\n"
            f"Dry run: {'yes - nothing will move' if self.state.dry_run else 'NO - THE ROBOT WILL MOVE'}.\n"
            "Approve only if the workspace is clear and the hardware e-stop is within reach."
        )

    def execute_precheck(self, plan_id: str) -> tuple[str | None, StoredPlan | None]:
        if not self.rate.would_allow("execute_plan"):
            return "rate limit reached for execute_plan", None
        reason = self.state.motion_blocked_reason
        if reason is not None:
            return reason, None
        if self.state.active is not None:
            return "another plan is executing", None
        try:
            stored = self.plans.peek(plan_id)
        except PlanError as e:
            return str(e), None
        if not stored.executable:
            return "plan was rejected by the safety envelope", stored
        return None, stored

    # --- execution ---------------------------------------------------------------------
    async def run_execution(self, stored: StoredPlan, ctx: Context, rec: _CallRecord) -> ExecutionReport:
        plan = stored.plan
        # If the motion starts in contact above the limit (typically: retreating after a force abort),
        # abort only if the force rises above its starting level, so the robot can back out.
        start_wrench = await self.backend.get_wrench()
        baseline_n = start_wrench.force_norm if start_wrench is not None else 0.0
        force_limit_n = max(self.policy.force.max_contact_force_n, baseline_n + _RETREAT_HYSTERESIS_N)
        start_torque = start_wrench.torque_norm if start_wrench is not None else 0.0
        torque_limit_nm = max(self.policy.force.max_contact_torque_nm, start_torque + _RETREAT_HYSTERESIS_NM)
        if start_wrench is None:
            rec.extra["force_monitoring"] = "unavailable"
            logger.warning("backend provides no wrench estimate: force limits cannot be monitored")
        active = ActiveExecution(plan_id=plan.plan_id, started_at=self.clock())
        self.state.active = active
        violation: list[Violation] = []

        async def on_progress(fraction: float, message: str) -> None:
            active.progress, active.message = fraction, message
            try:
                await ctx.report_progress(fraction, 1.0, message=message)
            except Exception:  # progress is best-effort; never let it break a motion
                logger.debug("progress report failed", exc_info=True)

        def should_abort() -> bool:
            return active.abort_reason is not None

        monitor_error: list[str] = []

        def over_limit(w: Wrench) -> list[Violation]:
            return [
                v
                for v in check_wrench(w, self.policy)
                if (v.code == "FORCE_LIMIT" and w.force_norm > force_limit_n)
                or (v.code == "TORQUE_LIMIT" and w.torque_norm > torque_limit_nm)
            ]

        async def monitor() -> None:
            # Fail-safe: if the wrench cannot be read, the motion is aborted.
            period = 1.0 / self.policy.force.monitor_rate_hz
            while True:
                try:
                    wrench = await self.backend.get_wrench()
                except Exception as e:
                    monitor_error.append(f"{type(e).__name__}: {e}")
                    active.request_abort("force monitor failed")
                    await self.backend.stop()
                    return
                if wrench is not None:
                    found = over_limit(wrench)
                    if found:
                        violation.extend(found)
                        active.request_abort(f"force monitor: {found[0].message}")
                        await self.backend.stop()
                        return
                await anyio.sleep(period)

        backend_error: BackendError | None = None
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(monitor)
                try:
                    result = await self.backend.execute(plan, on_progress, should_abort)
                except BackendError as e:  # re-raised below, outside the task group (no ExceptionGroup)
                    backend_error = e
                finally:
                    tg.cancel_scope.cancel()
        finally:
            self.state.active = None
        if backend_error is not None:
            self.state.last_result = f"{plan.plan_id}: failed ({backend_error})"
            raise backend_error
        if monitor_error:
            self.state.last_result = f"{plan.plan_id}: aborted (force monitor failed)"
            raise ToolError(f"execution of {plan.plan_id} aborted: force monitor failed ({monitor_error[0]})")

        if not violation and result.max_observed_force_n > force_limit_n:
            f = result.max_observed_force_n
            violation.append(
                Violation(
                    code="FORCE_LIMIT",
                    severity="hard",
                    message=f"contact force {f:.1f} N exceeded the {self.policy.force.max_contact_force_n} N limit",
                    detail={"force_n": f},
                )
            )
        final = result.final_joint_state.positions if result.final_joint_state else None
        final_pose = await self.backend.forward_kinematics(final) if final else None
        rec.extra["max_observed_force_n"] = round(result.max_observed_force_n, 3)
        if violation:
            v = violation[0]
            self.state.latch_force_violation(v)
            n = self.plans.invalidate_all("force-limit violation")
            self.audit.log(
                "force_violation", tool="execute_plan", plan_id=plan.plan_id, verdict=v, invalidated_plans=n
            )
            self.state.last_result = f"{plan.plan_id}: aborted by force monitor"
            raise ToolError(
                f"execution of {plan.plan_id} ABORTED: {v.message}. The violation is latched and the software "
                "e-stop is active; a human must inspect the scene and call reset_estop."
            )
        if active.abort_reason is not None:
            self.state.last_result = f"{plan.plan_id}: aborted ({active.abort_reason})"
            rec.outcome = "aborted"
            return ExecutionReport(
                plan_id=plan.plan_id,
                status="aborted",
                message=f"motion aborted: {active.abort_reason}",
                executed=True,
                dry_run=False,
                approval=str((rec.approval or {}).get("via", "?")),
                final_joint_positions=final,
                final_ee_pose=final_pose,
                max_observed_force_n=result.max_observed_force_n,
                duration_s=plan.duration_s,
            )
        if not result.success:
            self.state.last_result = f"{plan.plan_id}: failed ({result.message})"
            raise ToolError(f"execution of {plan.plan_id} failed: {result.message}")
        self.state.last_result = f"{plan.plan_id}: completed"
        return ExecutionReport(
            plan_id=plan.plan_id,
            status="completed",
            message=result.message,
            executed=True,
            dry_run=False,
            approval=str((rec.approval or {}).get("via", "?")),
            final_joint_positions=final,
            final_ee_pose=final_pose,
            max_observed_force_n=result.max_observed_force_n,
            duration_s=plan.duration_s,
        )

    # --- misc --------------------------------------------------------------------------
    def envelope_info(self) -> SafetyEnvelopeInfo:
        p = self.policy
        lines = [
            f"Robot {p.robot.name}; all positions in {p.robot.base_frame}, TCP frame {p.robot.ee_frame}."
        ]
        for name in p.robot.joint_names:
            lim = p.robot.joint_limits[name]
            lines.append(f"{name}: [{lim.min:+.4f}, {lim.max:+.4f}] rad, max {lim.max_velocity} rad/s")
        b = p.workspace.box
        lines.append(f"TCP must stay inside box min={list(b.min)} max={list(b.max)} m.")
        for z in p.workspace.keep_out:
            lines.append(f"Keep-out zone '{z.name}': min={list(z.min)} max={list(z.max)} m.")
        m = p.motion
        lines.append(
            f"Velocity scaling <= {m.max_velocity_scaling} (default {m.default_velocity_scaling}); acceleration "
            f"scaling <= {m.max_acceleration_scaling}. Any joint may travel at most {m.max_joint_step_rad} rad per "
            f"plan; Cartesian plans at most {m.max_cartesian_step_m} m of TCP path. Plans expire after "
            f"{m.plan_ttl_s:g} s and are single-use."
        )
        lines.append(
            f"Execution aborts and latches an e-stop if contact force > {p.force.max_contact_force_n} N or torque > "
            f"{p.force.max_contact_torque_nm} N*m."
        )
        if p.gripper:
            gp = p.gripper
            lines.append(
                f"Gripper width [{gp.min_width_m}, {gp.max_width_m}] m, grasp force <= {gp.max_grasp_force_n} N, "
                f"speed <= {gp.max_speed_mps} m/s."
            )
        lines.append(
            f"Approval mode: {p.approval.mode} (for {', '.join(p.approval.require_for)}); dry run: {self.state.dry_run}."
        )
        return SafetyEnvelopeInfo(
            summary="\n".join(lines),
            robot=p.robot.name,
            base_frame=p.robot.base_frame,
            ee_frame=p.robot.ee_frame,
            joint_names=p.robot.joint_names,
            joint_limits=p.robot.joint_limits,
            home_joint_positions=p.robot.home_joint_positions,
            workspace=p.workspace,
            motion=p.motion,
            force=p.force,
            gripper=p.gripper,
            controllers_allowlist=p.controllers.allowlist,
            camera_topics=p.perception.camera_topics,
            max_image_width=p.perception.max_image_width,
            approval=p.approval,
            rate_limits=p.rate_limits,
            dry_run=self.state.dry_run,
            enabled_tool_groups=list(p.tools.enabled),
        )


# --------------------------------------------------------------------------------------
# MCP wiring
# --------------------------------------------------------------------------------------
_RO = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True)
_RO_LOCAL = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
_PLAN = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=False, open_world_hint=True
)
_ACT = ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=True
)
_STOP = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
)


@dataclass
class ArmGuardApp:
    server: MCPServer
    guard: ArmGuard


def build(
    policy: Policy,
    backend: RobotBackend,
    audit: AuditLogger | None = None,
    *,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    request_state_ttl_s: float = 300.0,
) -> ArmGuardApp:
    """Build the MCP server and its :class:`ArmGuard` core. Only enabled tool groups are registered."""
    guard = ArmGuard(policy, backend, audit, clock=clock, monotonic=monotonic)
    server = MCPServer(
        SERVER_NAME,
        title="armguard-mcp: safety-first MCP server for ROS 2 manipulators",
        version=__version__,
        instructions=INSTRUCTIONS,
        request_state_security=RequestStateSecurity.ephemeral(ttl=request_state_ttl_s),
    )
    enabled = set(policy.tools.enabled) | {"safety"}
    if "introspect" in enabled:
        _register_introspect(server, guard)
    if "perception" in enabled:
        _register_perception(server, guard)
    if "motion" in enabled:
        _register_motion(server, guard)
    if "gripper" in enabled:
        _register_gripper(server, guard)
    if "control" in enabled:
        _register_control(server, guard)
    _register_safety(server, guard)
    guard.audit.log(
        "server_start", version=__version__, backend=backend.name, tools=sorted(_tool_names(enabled))
    )
    return ArmGuardApp(server=server, guard=guard)


def build_server(
    policy: Policy,
    backend: RobotBackend,
    audit: AuditLogger | None = None,
    *,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
) -> MCPServer:
    return build(policy, backend, audit, clock=clock, monotonic=monotonic).server


def _tool_names(groups: set[str]) -> set[str]:
    return {t for gname in groups for t in TOOL_GROUPS[gname]}


def _register_introspect(server: MCPServer, guard: ArmGuard) -> None:
    @server.tool(annotations=_RO)
    async def get_robot_state() -> RobotState:
        """Current joint positions [rad], TCP pose [m, quaternion xyzw] in the robot base frame, estimated
        external wrench [N, N*m], gripper state and safety status. Read-only."""
        async with guard.call("get_robot_state"):
            b = guard.backend
            return RobotState(
                robot=guard.policy.robot.name,
                backend=b.name,
                joint_state=await b.get_joint_state(),
                ee_pose=await b.get_ee_pose(),
                wrench=await b.get_wrench(),
                gripper=await b.get_gripper_state(),
                safety=guard.state.status(),
            )

    @server.tool(annotations=_RO_LOCAL)
    async def get_safety_envelope() -> SafetyEnvelopeInfo:
        """The limits the server enforces (joint limits, workspace box, keep-out zones, speed/step caps,
        force thresholds, gripper limits, approval mode). Read this before planning and stay inside it:
        violations are rejected server-side and cannot be overridden."""
        async with guard.call("get_safety_envelope"):
            return guard.envelope_info()

    @server.tool(annotations=_RO)
    async def lookup_transform(target_frame: str, source_frame: str) -> Pose:
        """Pose of source_frame expressed in target_frame (tf2 semantics), e.g. target_frame='fr3_link0',
        source_frame='fr3_hand_tcp'. Position in metres, orientation as a quaternion (x, y, z, w)."""
        async with guard.call(
            "lookup_transform", {"target_frame": target_frame, "source_frame": source_frame}
        ):
            return await guard.backend.lookup_transform(target_frame, source_frame)

    @server.tool(annotations=_RO)
    async def list_controllers() -> ControllerList:
        """ros2_control controllers with their type and state, plus which ones the policy lets you switch."""
        async with guard.call("list_controllers"):
            return ControllerList(
                controllers=await guard.backend.list_controllers(),
                switchable=guard.policy.controllers.allowlist,
            )

    @server.tool(annotations=_RO)
    async def list_ros_graph() -> GraphInfo:
        """ROS 2 nodes, topics, services and actions visible to the server. Read-only; you cannot publish
        or call arbitrary services through this server."""
        async with guard.call("list_ros_graph"):
            return await guard.backend.list_graph()

    @server.tool(annotations=_RO_LOCAL)
    async def get_audit_tail(n: Annotated[int, Field(ge=1, le=200)] = 20) -> AuditTail:
        """The last n audit-log events (tool calls, approvals, denials, e-stops). Images are redacted."""
        async with guard.call("get_audit_tail", {"n": n}):
            return AuditTail(events=guard.audit.tail(n))


def _register_perception(server: MCPServer, guard: ArmGuard) -> None:
    @server.tool(annotations=_RO)
    async def camera_snapshot(topic: str) -> CallToolResult:
        """Grab the latest image from an allowlisted camera topic (see get_safety_envelope.camera_topics).
        Returns the image plus JSON metadata (width, height, stamp). Images wider than the policy's
        max_image_width are downscaled (or refused if Pillow is not installed)."""
        async with guard.call("camera_snapshot", {"topic": topic}) as rec:
            pc = guard.policy.perception
            if topic not in pc.camera_topics:
                guard.deny(f"camera topic {topic!r} is not allowlisted; allowed: {pc.camera_topics}")
            frame = await guard.backend.camera_snapshot(topic, pc.max_image_width)
            data, mime, w, h = frame.data, frame.mime, frame.width, frame.height
            downscaled = False
            if w > pc.max_image_width:
                if not pillow_available():
                    guard.deny(
                        f"image is {w}px wide (policy max_image_width={pc.max_image_width}) and Pillow is not "
                        "installed for downscaling; install armguard-mcp[image]"
                    )
                data, mime, w, h = downscale(data, pc.max_image_width)
                downscaled = True
            meta = {
                "topic": topic,
                "mime_type": mime,
                "width": w,
                "height": h,
                "stamp": frame.stamp,
                "downscaled": downscaled,
                "bytes": len(data),
            }
            rec.extra["image"] = {k: v for k, v in meta.items() if k != "bytes"} | {"size_bytes": len(data)}
            return CallToolResult(
                content=[
                    ImageContent(type="image", data=base64.b64encode(data).decode(), mime_type=mime),
                    TextContent(type="text", text=json.dumps(meta)),
                ]
            )


def _register_motion(server: MCPServer, guard: ArmGuard) -> None:
    def _plan_args(**kw: Any) -> dict[str, Any]:
        return {
            k: (v.model_dump() if isinstance(v, BaseModel) else v) for k, v in kw.items() if v is not None
        }

    @server.tool(annotations=_PLAN)
    async def plan_to_joints(
        joint_positions: Annotated[
            list[float], Field(description="Target joint positions [rad], one per joint")
        ],
        velocity_scaling: Annotated[
            float | None, Field(description="Fraction of max joint speed (0, 1]")
        ] = None,
        acceleration_scaling: Annotated[
            float | None, Field(description="Fraction of max joint accel (0, 1]")
        ] = None,
    ) -> PlanSummary:
        """Plan (do NOT move) a joint-space motion from the current state to joint_positions [rad].
        Returns a plan summary validated against the safety envelope; if status is not 'rejected',
        pass plan_id to execute_plan. Scaling above the policy cap is clamped."""
        args = _plan_args(
            joint_positions=joint_positions,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        )
        async with guard.call("plan_to_joints", args) as rec:
            guard.require_motion_allowed()
            n = len(guard.policy.robot.joint_names)
            if len(joint_positions) != n:
                guard.deny(
                    f"expected {n} joint positions ({guard.policy.robot.joint_names}), got {len(joint_positions)}"
                )
            if not all(math.isfinite(v) for v in joint_positions):
                guard.deny("joint positions must be finite numbers")
            vs, as_, notes = guard.scaling(velocity_scaling, acceleration_scaling)
            plan = await guard.backend.plan_to_joints(joint_positions, vs, as_)
            summary = await guard.finalize_plan(plan, notes)
            rec.plan_id, rec.verdict = plan.plan_id, summary.status
            return summary

    @server.tool(annotations=_PLAN)
    async def plan_to_pose(
        position: Annotated[Vector3, Field(description="Target TCP position [m]")],
        orientation: Annotated[
            Quaternion | None, Field(description="Target TCP orientation (x, y, z, w); omit to keep current")
        ] = None,
        frame_id: Annotated[
            str | None, Field(description="Frame of the target; default = robot base frame")
        ] = None,
        velocity_scaling: Annotated[
            float | None, Field(description="Fraction of max joint speed (0, 1]")
        ] = None,
        acceleration_scaling: Annotated[
            float | None, Field(description="Fraction of max joint accel (0, 1]")
        ] = None,
    ) -> PlanSummary:
        """Plan (do NOT move) a motion that brings the TCP to a pose. Uses inverse kinematics; the path is
        a joint-space interpolation (the TCP does not move in a straight line - use plan_cartesian_path
        for that). Returns a plan summary validated against the safety envelope."""
        args = _plan_args(
            position=position,
            orientation=orientation,
            frame_id=frame_id,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        )
        async with guard.call("plan_to_pose", args) as rec:
            guard.require_motion_allowed()
            vs, as_, notes = guard.scaling(velocity_scaling, acceleration_scaling)
            target = await guard.to_base_frame(position, orientation, frame_id)
            plan = await guard.backend.plan_to_pose(target, vs, as_)
            summary = await guard.finalize_plan(plan, notes)
            rec.plan_id, rec.verdict = plan.plan_id, summary.status
            return summary

    @server.tool(annotations=_PLAN)
    async def plan_cartesian_path(
        waypoints: Annotated[
            list[CartesianWaypoint], Field(min_length=1, max_length=50, description="TCP waypoints, in order")
        ],
        frame_id: Annotated[
            str | None, Field(description="Frame of the waypoints; default = base frame")
        ] = None,
        velocity_scaling: Annotated[
            float | None, Field(description="Fraction of max joint speed (0, 1]")
        ] = None,
        acceleration_scaling: Annotated[
            float | None, Field(description="Fraction of max joint accel (0, 1]")
        ] = None,
    ) -> PlanSummary:
        """Plan (do NOT move) a straight-line TCP path through the waypoints [m]. Total TCP path length is
        capped by the policy (max_cartesian_step_m). Returns a plan summary validated against the envelope."""
        args = _plan_args(
            waypoints=[w.model_dump() for w in waypoints],
            frame_id=frame_id,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        )
        async with guard.call("plan_cartesian_path", args) as rec:
            guard.require_motion_allowed()
            vs, as_, notes = guard.scaling(velocity_scaling, acceleration_scaling)
            poses = [await guard.to_base_frame(w.position, w.orientation, frame_id) for w in waypoints]
            plan = await guard.backend.plan_cartesian(
                poses, guard.policy.motion.cartesian_eef_step_m, vs, as_
            )
            summary = await guard.finalize_plan(plan, notes)
            rec.plan_id, rec.verdict = plan.plan_id, summary.status
            return summary

    async def _approve_execute(plan_id: str, ctx: Context) -> PolicyDecision | Elicit[ApprovalForm]:
        # Pure: may run more than once per call under protocol 2026-07-28.
        precheck, stored = guard.execute_precheck(plan_id)
        if precheck is None and stored is not None:
            js = await guard.backend.get_joint_state()
            if max_deviation(js.positions, stored.plan.start) > guard.policy.motion.start_tolerance_rad:
                precheck = "stale plan: the robot moved since planning"
        if precheck is not None or stored is None:
            return guard.approval_request(
                ctx, required=True, message="", precheck_error=precheck or "unknown plan"
            )
        required = (
            guard.approval_required("execute", stored.verdict.inside_envelope) and not guard.state.dry_run
        )
        return guard.approval_request(ctx, required=required, message=guard.execute_message(stored))

    @server.tool(annotations=_ACT)
    async def execute_plan(
        plan_id: Annotated[str, Field(description="plan_id returned by a plan_* tool")],
        ctx: Context,
        approval: Annotated[ApprovalOutcome, Resolve(_approve_execute)],
    ) -> ExecutionReport:
        """EXECUTE a previously planned motion on the robot. The server re-validates the plan against the
        safety envelope, rejects stale/expired/used plans, may ask a human to approve (depending on
        policy), monitors contact force while moving (aborting and latching an e-stop above the limit),
        and streams progress. Plan handles are single-use. In dry-run mode nothing moves."""
        async with guard.call("execute_plan", {"plan_id": plan_id}) as rec:
            rec.plan_id = plan_id
            guard.require_motion_allowed()
            if guard.state.active is not None:
                guard.deny(
                    f"refused: plan {guard.state.active.plan_id} is still executing; call stop_motion first"
                )
            stored = guard.plans.peek(plan_id)
            rec.verdict = stored.verdict.model_dump()
            if not stored.executable:
                guard.plans.consume(plan_id, "rejected by the safety envelope")
                reasons = "; ".join(v.message for v in stored.verdict.hard)
                guard.deny(
                    f"refused: plan {plan_id} violates hard safety limits and can never be executed: {reasons}"
                )
            verdict, _, _ = await guard.validate_plan(stored.plan)  # independent re-check
            if not verdict.ok:
                guard.plans.consume(plan_id, "rejected by the safety envelope")
                guard.deny(f"refused: re-validation failed: {'; '.join(v.message for v in verdict.hard)}")
            js = await guard.backend.get_joint_state()
            dev = max_deviation(js.positions, stored.plan.start)
            if dev > guard.policy.motion.start_tolerance_rad:
                guard.plans.consume(plan_id, "stale (robot moved after planning)")
                guard.deny(
                    f"refused: stale plan - the robot moved {dev:.4f} rad since plan {plan_id} was made "
                    f"(tolerance {guard.policy.motion.start_tolerance_rad} rad). Plan again from the current state."
                )
            dry = guard.state.dry_run
            required = guard.approval_required("execute", verdict.inside_envelope) and not dry
            try:
                guard.check_approval(approval, required, rec)
            except Denied:
                guard.plans.consume(plan_id, "approval denied")
                raise
            guard.plans.consume(plan_id)
            if dry:
                rec.outcome = "dry_run"
                return ExecutionReport(
                    plan_id=plan_id,
                    status="dry_run",
                    message="dry run: plan is valid and would have been executed"
                    + (
                        " (would need human approval)"
                        if guard.approval_required("execute", verdict.inside_envelope)
                        else ""
                    ),
                    executed=False,
                    dry_run=True,
                    approval=str((rec.approval or {}).get("via", "?")),
                    final_joint_positions=stored.summary.final_joint_positions,
                    final_ee_pose=stored.summary.final_ee_pose,
                    duration_s=stored.plan.duration_s,
                )
            return await guard.run_execution(stored, ctx, rec)

    @server.tool(annotations=_RO_LOCAL)
    async def get_motion_status() -> MotionStatus:
        """Whether a plan is executing, its progress (0..1) and the result of the last execution."""
        async with guard.call("get_motion_status"):
            return guard.state.motion_status()


def _register_gripper(server: MCPServer, guard: ArmGuard) -> None:
    gp = guard.policy.gripper
    assert gp is not None

    def _check_width(width: float, what: str = "width_m") -> None:
        if not (math.isfinite(width) and gp.min_width_m <= width <= gp.max_width_m):
            guard.deny(
                f"{what}={width} m is outside the allowed range [{gp.min_width_m}, {gp.max_width_m}] m"
            )

    def _speed(speed: float | None) -> float:
        if speed is None:
            return gp.max_speed_mps
        if not (math.isfinite(speed) and 0 < speed <= gp.max_speed_mps):
            guard.deny(f"speed_mps={speed} is outside (0, {gp.max_speed_mps}] m/s")
        return speed

    @server.tool(annotations=_ACT)
    async def gripper_move(
        width_m: Annotated[float, Field(description="Target finger opening [m]")],
        speed_mps: Annotated[
            float | None, Field(description="Finger speed [m/s]; default = policy max")
        ] = None,
    ) -> ActionResult:
        """Move the gripper fingers to an opening width [m] without applying grasp force."""
        async with guard.call("gripper_move", {"width_m": width_m, "speed_mps": speed_mps}):
            guard.require_motion_allowed()
            _check_width(width_m)
            st = await guard.backend.gripper_move(width_m, _speed(speed_mps))
            return ActionResult(ok=True, message=f"gripper at {st.width_m:.4f} m", data=st.model_dump())

    @server.tool(annotations=_ACT)
    async def gripper_grasp(
        width_m: Annotated[float, Field(description="Expected object width [m]")],
        force_n: Annotated[float, Field(description="Grasp force [N]; must not exceed the policy maximum")],
        speed_mps: Annotated[
            float | None, Field(description="Finger speed [m/s]; default = policy max")
        ] = None,
        epsilon_inner_m: Annotated[float, Field(ge=0, le=0.05)] = 0.005,
        epsilon_outer_m: Annotated[float, Field(ge=0, le=0.05)] = 0.005,
    ) -> ActionResult:
        """Close the gripper on an object of about width_m [m] with force_n [N]. Succeeds (ok=true) only
        if the fingers stop within [width - epsilon_inner, width + epsilon_outer]. Forces above the
        policy maximum are rejected, not clamped."""
        args = {"width_m": width_m, "force_n": force_n, "speed_mps": speed_mps}
        async with guard.call("gripper_grasp", args):
            guard.require_motion_allowed()
            _check_width(width_m)
            if not (math.isfinite(force_n) and 0 < force_n <= gp.max_grasp_force_n):
                guard.deny(
                    f"force_n={force_n} N rejected: must be in (0, {gp.max_grasp_force_n}] N (policy maximum)"
                )
            st = await guard.backend.gripper_grasp(
                width_m, force_n, _speed(speed_mps), epsilon_inner_m, epsilon_outer_m
            )
            msg = "object grasped" if st.is_grasped else "grasp failed: no object within the width tolerance"
            return ActionResult(ok=st.is_grasped, message=msg, data=st.model_dump())

    @server.tool(annotations=_ACT)
    async def gripper_home() -> ActionResult:
        """Home (fully open and calibrate) the gripper."""
        async with guard.call("gripper_home"):
            guard.require_motion_allowed()
            st = await guard.backend.gripper_home()
            return ActionResult(ok=True, message="gripper homed", data=st.model_dump())


def _register_control(server: MCPServer, guard: ArmGuard) -> None:
    def _switch_precheck(activate: list[str], deactivate: list[str]) -> str | None:
        allow = set(guard.policy.controllers.allowlist)
        bad = sorted({*activate, *deactivate} - allow)
        if bad:
            return f"controller(s) {bad} are not in the policy allowlist {sorted(allow)}"
        if not activate and not deactivate:
            return "nothing to switch: pass activate and/or deactivate"
        both = sorted(set(activate) & set(deactivate))
        if both:
            return f"controller(s) {both} are in both activate and deactivate"
        if guard.state.active is not None:
            return "cannot switch controllers while a plan is executing"
        return guard.state.motion_blocked_reason

    async def _approve_switch(
        activate: list[str], deactivate: list[str], ctx: Context
    ) -> PolicyDecision | Elicit[ApprovalForm]:
        pre = _switch_precheck(activate, deactivate)
        if pre is None and not guard.rate.would_allow("switch_controllers"):
            pre = "rate limit reached"
        msg = (
            f"APPROVE CONTROLLER SWITCH on '{guard.policy.robot.name}'? activate={activate} deactivate={deactivate}. "
            "Switching controllers changes how the arm responds to commands and contact."
        )
        required = guard.approval_required("switch_controllers")
        return guard.approval_request(ctx, required=required, message=msg, precheck_error=pre)

    @server.tool(annotations=_ACT)
    async def switch_controllers(
        ctx: Context,
        approval: Annotated[ApprovalOutcome, Resolve(_approve_switch)],
        activate: Annotated[
            list[str], Field(description="Controllers to activate (must be allowlisted)")
        ] = [],  # noqa: B006
        deactivate: Annotated[
            list[str], Field(description="Controllers to deactivate (must be allowlisted)")
        ] = [],  # noqa: B006
    ) -> ControllerList:
        """Activate/deactivate ros2_control controllers. Only controllers in the policy allowlist may be
        switched; a human may need to approve."""
        async with guard.call("switch_controllers", {"activate": activate, "deactivate": deactivate}) as rec:
            guard.require_motion_allowed()
            pre = _switch_precheck(activate, deactivate)
            if pre is not None:
                guard.deny(f"refused: {pre}")
            guard.check_approval(approval, guard.approval_required("switch_controllers"), rec)
            if guard.state.dry_run:
                rec.outcome = "dry_run"
                return ControllerList(
                    controllers=await guard.backend.list_controllers(),
                    switchable=guard.policy.controllers.allowlist,
                )
            controllers = await guard.backend.switch_controllers(activate, deactivate)
            return ControllerList(controllers=controllers, switchable=guard.policy.controllers.allowlist)

    def _thresholds_precheck(force_n: float, torque_nm: float) -> str | None:
        f = guard.policy.force
        if not (math.isfinite(force_n) and 0 < force_n <= f.max_contact_force_n):
            return (
                f"force_n={force_n} N must be in (0, {f.max_contact_force_n}] N (policy max_contact_force_n)"
            )
        if not (math.isfinite(torque_nm) and 0 < torque_nm <= f.max_contact_torque_nm):
            return f"torque_nm={torque_nm} N*m must be in (0, {f.max_contact_torque_nm}] N*m (policy maximum)"
        if guard.state.active is not None:
            return "cannot change collision thresholds while a plan is executing"
        return guard.state.motion_blocked_reason

    async def _approve_thresholds(
        force_n: float, torque_nm: float, ctx: Context
    ) -> PolicyDecision | Elicit[ApprovalForm]:
        pre = _thresholds_precheck(force_n, torque_nm)
        msg = (
            f"APPROVE COLLISION THRESHOLD CHANGE on '{guard.policy.robot.name}'? force={force_n} N, "
            f"torque={torque_nm} N*m (policy maxima {guard.policy.force.max_contact_force_n} N / "
            f"{guard.policy.force.max_contact_torque_nm} N*m)."
        )
        required = guard.approval_required("set_collision_thresholds")
        return guard.approval_request(ctx, required=required, message=msg, precheck_error=pre)

    @server.tool(annotations=_ACT)
    async def set_collision_thresholds(
        force_n: Annotated[
            float, Field(description="Collision force threshold [N]; <= policy max_contact_force_n")
        ],
        torque_nm: Annotated[float, Field(description="Collision torque threshold [N*m]; <= policy maximum")],
        ctx: Context,
        approval: Annotated[ApprovalOutcome, Resolve(_approve_thresholds)],
    ) -> ActionResult:
        """Set the robot's collision (reflex) thresholds. Values above the policy maxima are rejected."""
        async with guard.call(
            "set_collision_thresholds", {"force_n": force_n, "torque_nm": torque_nm}
        ) as rec:
            guard.require_motion_allowed()
            pre = _thresholds_precheck(force_n, torque_nm)
            if pre is not None:
                guard.deny(f"refused: {pre}")
            guard.check_approval(approval, guard.approval_required("set_collision_thresholds"), rec)
            if guard.state.dry_run:
                rec.outcome = "dry_run"
                return ActionResult(ok=True, message="dry run: thresholds not changed")
            await guard.backend.set_collision_thresholds(force_n, torque_nm)
            return ActionResult(ok=True, message=f"collision thresholds set to {force_n} N / {torque_nm} N*m")


def _register_safety(server: MCPServer, guard: ArmGuard) -> None:
    @server.tool(annotations=_STOP)
    async def stop_motion() -> ActionResult:
        """Stop the current motion immediately. Always available, never rate limited, never needs approval.
        Does not latch; use estop to also block further motion."""
        async with guard.call("stop_motion"):
            active = guard.state.active
            if active is not None:
                active.request_abort("stop_motion")
            await guard.backend.stop()
            msg = (
                f"stop requested for plan {active.plan_id}"
                if active
                else "no motion in progress; stop sent anyway"
            )
            return ActionResult(ok=True, message=msg)

    @server.tool(annotations=_STOP)
    async def estop(reason: Annotated[str, Field(max_length=500)] = "requested by agent") -> SafetyStatus:
        """SOFTWARE E-STOP: stop all motion, invalidate every plan and refuse motion/gripper/control tools
        until a human approves reset_estop. Always available, never rate limited, never needs approval.
        This is not a substitute for the hardware e-stop."""
        async with guard.call("estop", {"reason": reason}):
            guard.state.estop(reason)
            n = guard.plans.invalidate_all("e-stop")
            try:
                await guard.backend.stop()
            finally:
                guard.audit.log("estop", tool="estop", reason=reason, invalidated_plans=n)
            return guard.state.status()

    async def _approve_reset(ctx: Context) -> PolicyDecision | Elicit[ApprovalForm]:
        s = guard.state
        if not s.estopped:
            return guard.approval_request(ctx, required=False, message="")
        if not guard.rate.would_allow("reset_estop"):
            return guard.approval_request(ctx, required=True, message="", precheck_error="rate limit reached")
        msg = (
            f"RESET SOFTWARE E-STOP on '{guard.policy.robot.name}'? It was triggered because: {s.reason}. "
            + ("A FORCE-LIMIT VIOLATION was latched. " if s.force_violation else "")
            + "Only approve after inspecting the robot and its surroundings. Plans made before the e-stop stay invalid."
        )
        return guard.approval_request(ctx, required=True, message=msg)

    @server.tool(
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
    )
    async def reset_estop(
        ctx: Context, approval: Annotated[ApprovalOutcome, Resolve(_approve_reset)]
    ) -> SafetyStatus:
        """Release the software e-stop (and any latched force violation). ALWAYS requires human approval via
        the MCP client, regardless of the approval mode."""
        async with guard.call("reset_estop") as rec:
            if not guard.state.estopped:
                rec.outcome = "noop"
                return guard.state.status()
            guard.check_approval(approval, True, rec)
            guard.state.reset()
            guard.audit.log("estop_reset", tool="reset_estop", approval=rec.approval)
            return guard.state.status()

    async def _approve_recovery(ctx: Context) -> PolicyDecision | Elicit[ApprovalForm]:
        pre = guard.state.motion_blocked_reason
        if pre is None and guard.state.active is not None:
            pre = "a plan is executing"
        msg = (
            f"APPROVE ERROR RECOVERY on '{guard.policy.robot.name}'? This clears the robot's reflex/error state so it "
            "can accept commands again. Make sure the cause (e.g. a collision) has been resolved."
        )
        return guard.approval_request(
            ctx, required=guard.approval_required("error_recovery"), message=msg, precheck_error=pre
        )

    @server.tool(
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
    )
    async def error_recovery(
        ctx: Context, approval: Annotated[ApprovalOutcome, Resolve(_approve_recovery)]
    ) -> ActionResult:
        """Clear the robot's error/reflex state (e.g. after a collision reflex). May require human approval.
        Refused while the software e-stop is active (reset it first)."""
        async with guard.call("error_recovery") as rec:
            guard.require_motion_allowed()
            if guard.state.active is not None:
                guard.deny("refused: a plan is executing")
            guard.check_approval(approval, guard.approval_required("error_recovery"), rec)
            if guard.state.dry_run:
                rec.outcome = "dry_run"
                return ActionResult(ok=True, message="dry run: error recovery not sent")
            await guard.backend.error_recovery()
            return ActionResult(ok=True, message="error recovery complete")

    @server.tool(annotations=_RO_LOCAL)
    async def get_safety_status() -> SafetyStatus:
        """E-stop state, latched force violation, dry-run flag and the last envelope violation. Never rate limited."""
        async with guard.call("get_safety_status"):
            return guard.state.status()
