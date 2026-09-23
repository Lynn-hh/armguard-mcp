"""Mutable safety state: software e-stop latch, force-violation latch, dry-run flag, motion status."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from armguard_mcp.models import MotionStatus, SafetyStatus, Violation
from armguard_mcp.safety.audit import iso_utc

EstopSource = Literal["agent", "server"]


@dataclass
class ActiveExecution:
    """The one arm execution the server allows at a time.

    It is claimed synchronously at the top of ``execute_plan`` (before any ``await``), so a
    concurrent ``execute_plan`` is refused and ``stop_motion`` / ``estop`` always find it, even
    while the plan is still being re-validated and nothing has been sent to the robot yet.
    """

    plan_id: str
    started_at: float
    progress: float = 0.0
    message: str = "validating"
    abort_reason: str | None = None

    def request_abort(self, reason: str) -> None:
        if self.abort_reason is None:
            self.abort_reason = reason


@dataclass
class ActiveGripper:
    """A gripper action in progress. ``estop`` / ``stop_motion`` abort it and cancel ``scope``."""

    action: str
    abort_reason: str | None = None
    scope: Any = None  # anyio.CancelScope of the awaiting tool call

    def request_abort(self, reason: str) -> None:
        if self.abort_reason is None:
            self.abort_reason = reason


@dataclass
class SafetyState:
    """Latched safety state shared by all tool calls.

    * ``estop(reason)`` latches: every motion / gripper / control tool is refused until
      ``reset()`` (exposed as the ``reset_estop`` tool, which always needs human approval).
      Every latch gets a new ``estop_event`` number, so an approval shown for one e-stop cannot
      release a later one.
    * A force/torque violation during execution latches the e-stop *and* records the violation.
    """

    dry_run: bool = False
    clock: Callable[[], float] = time.time
    estopped: bool = False
    reason: str | None = None
    reason_source: EstopSource | None = None
    estopped_at: float | None = None
    estop_event: int = 0
    force_violation: Violation | None = None
    last_violation: Violation | None = None
    active: ActiveExecution | None = None
    gripper: ActiveGripper | None = None
    last_result: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --- e-stop -------------------------------------------------------------------------
    def estop(self, reason: str, source: EstopSource = "server") -> int:
        """Latch the e-stop (a new event every call) and abort whatever is moving. Returns the event."""
        with self._lock:
            if not self.estopped:
                self.estopped_at = self.clock()
            self.estopped = True
            self.reason = reason
            self.reason_source = source
            self.estop_event += 1
            event = self.estop_event
            active, gripper = self.active, self.gripper
        if active is not None:
            active.request_abort(f"e-stop: {reason}")
        if gripper is not None:
            gripper.request_abort(f"e-stop: {reason}")
        return event

    def latch_force_violation(self, violation: Violation) -> None:
        with self._lock:
            self.force_violation = violation
            self.last_violation = violation
        self.estop(f"force limit: {violation.message}", source="server")

    def record_violation(self, violation: Violation) -> None:
        with self._lock:
            self.last_violation = violation

    def reset(self, event: int | None = None) -> bool:
        """Release the latch. With ``event``, only if no newer e-stop was latched since; returns success."""
        with self._lock:
            if event is not None and event != self.estop_event:
                return False
            self.estopped = False
            self.reason = None
            self.reason_source = None
            self.estopped_at = None
            self.force_violation = None
            return True

    @property
    def motion_blocked_reason(self) -> str | None:
        if self.force_violation is not None:
            return f"force-limit violation latched ({self.force_violation.message}); call reset_estop"
        if self.estopped:
            return f"software e-stop is active ({self.reason}); call reset_estop (requires human approval)"
        return None

    # --- execution slot -----------------------------------------------------------------
    def claim_execution(self, plan_id: str, started_at: float) -> ActiveExecution | None:
        """Atomically take the single execution slot; None if another execution holds it."""
        with self._lock:
            if self.active is not None:
                return None
            self.active = ActiveExecution(plan_id=plan_id, started_at=started_at)
            return self.active

    def release_execution(self, active: ActiveExecution) -> None:
        """Free the slot, but only if ``active`` still owns it."""
        with self._lock:
            if self.active is active:
                self.active = None

    def claim_gripper(self, action: str) -> ActiveGripper | None:
        with self._lock:
            if self.gripper is not None:
                return None
            self.gripper = ActiveGripper(action=action)
            return self.gripper

    def release_gripper(self, gripper: ActiveGripper) -> None:
        with self._lock:
            if self.gripper is gripper:
                self.gripper = None

    def abort_all(self, reason: str) -> tuple[ActiveExecution | None, ActiveGripper | None]:
        """Request an abort of the arm execution and the gripper action (``stop_motion``)."""
        with self._lock:
            active, gripper = self.active, self.gripper
        if active is not None:
            active.request_abort(reason)
        if gripper is not None:
            gripper.request_abort(reason)
        return active, gripper

    # --- reporting ----------------------------------------------------------------------
    def status(self) -> SafetyStatus:
        return SafetyStatus(
            estopped=self.estopped,
            reason=self.reason,
            reason_source=self.reason_source if self.estopped else None,
            estop_event=self.estop_event if self.estopped else None,
            estopped_at=iso_utc(self.estopped_at) if self.estopped_at is not None else None,
            dry_run=self.dry_run,
            force_violation_latched=self.force_violation is not None,
            last_violation=self.last_violation,
            executing_plan_id=self.active.plan_id if self.active else None,
        )

    def motion_status(self) -> MotionStatus:
        a = self.active
        if a is None:
            return MotionStatus(executing=False, last_result=self.last_result)
        return MotionStatus(
            executing=True,
            plan_id=a.plan_id,
            progress=a.progress,
            message=a.message,
            last_result=self.last_result,
        )
