"""Mutable safety state: software e-stop latch, force-violation latch, dry-run flag, motion status."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from armguard_mcp.models import MotionStatus, SafetyStatus, Violation
from armguard_mcp.safety.audit import iso_utc


@dataclass
class ActiveExecution:
    plan_id: str
    started_at: float
    progress: float = 0.0
    message: str = ""
    abort_reason: str | None = None

    def request_abort(self, reason: str) -> None:
        if self.abort_reason is None:
            self.abort_reason = reason


@dataclass
class SafetyState:
    """Latched safety state shared by all tool calls.

    * ``estop(reason)`` latches: every motion / gripper / control tool is refused until
      ``reset()`` (exposed as the ``reset_estop`` tool, which always needs human approval).
    * A force/torque violation during execution latches the e-stop *and* records the violation.
    """

    dry_run: bool = False
    clock: Callable[[], float] = time.time
    estopped: bool = False
    reason: str | None = None
    estopped_at: float | None = None
    force_violation: Violation | None = None
    last_violation: Violation | None = None
    active: ActiveExecution | None = None
    last_result: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def estop(self, reason: str) -> None:
        with self._lock:
            if not self.estopped:
                self.estopped_at = self.clock()
            self.estopped = True
            self.reason = reason
            if self.active is not None:
                self.active.request_abort(f"e-stop: {reason}")

    def latch_force_violation(self, violation: Violation) -> None:
        with self._lock:
            self.force_violation = violation
            self.last_violation = violation
        self.estop(f"force limit: {violation.message}")

    def record_violation(self, violation: Violation) -> None:
        with self._lock:
            self.last_violation = violation

    def reset(self) -> None:
        with self._lock:
            self.estopped = False
            self.reason = None
            self.estopped_at = None
            self.force_violation = None

    @property
    def motion_blocked_reason(self) -> str | None:
        if self.force_violation is not None:
            return f"force-limit violation latched ({self.force_violation.message}); call reset_estop"
        if self.estopped:
            return f"software e-stop is active ({self.reason}); call reset_estop (requires human approval)"
        return None

    def status(self) -> SafetyStatus:
        return SafetyStatus(
            estopped=self.estopped,
            reason=self.reason,
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
