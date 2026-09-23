"""Plan handles: TTL-limited, single-use, bound to the robot state at planning time."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from armguard_mcp.models import EnvelopeVerdict, Plan, PlanSummary


class PlanError(Exception):
    """Base class: the plan handle cannot be used. Message is safe to show to the LLM."""


class PlanNotFound(PlanError):
    pass


class PlanExpired(PlanError):
    pass


class PlanAlreadyUsed(PlanError):
    pass


class PlanInvalidated(PlanError):
    pass


def new_plan_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class StoredPlan:
    plan: Plan
    summary: PlanSummary
    verdict: EnvelopeVerdict
    expires_at: float

    @property
    def executable(self) -> bool:
        return self.verdict.ok


def max_deviation(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)


class PlanStore:
    """Holds plans between ``plan_*`` and ``execute_plan``.

    Rejected plans are kept (non-executable) so ``execute_plan`` can explain *why* it refuses.
    A plan id can be consumed exactly once; all plans are invalidated on e-stop.
    """

    def __init__(self, ttl_s: float, clock: Callable[[], float] = time.time, max_plans: int = 256) -> None:
        self.ttl_s = ttl_s
        self._clock = clock
        self._max = max_plans
        self._plans: dict[str, StoredPlan] = {}
        self._gone: dict[str, str] = {}  # plan_id -> reason it can no longer be used
        self._lock = threading.Lock()

    def put(
        self, plan: Plan, summary: PlanSummary, verdict: EnvelopeVerdict, expires_at: float | None = None
    ) -> StoredPlan:
        with self._lock:
            self._purge_locked()
            exp = expires_at if expires_at is not None else self._clock() + self.ttl_s
            stored = StoredPlan(plan=plan, summary=summary, verdict=verdict, expires_at=exp)
            self._plans[plan.plan_id] = stored
            while len(self._plans) > self._max:
                oldest = next(iter(self._plans))
                self._plans.pop(oldest)
                self._gone[oldest] = "evicted (too many outstanding plans)"
            return stored

    def next_expiry(self) -> float:
        """Expiry time a plan stored right now would get."""
        return self._clock() + self.ttl_s

    def peek(self, plan_id: str) -> StoredPlan:
        """Look up without consuming. Raises a :class:`PlanError` subclass if unusable."""
        with self._lock:
            return self._get_locked(plan_id)

    def consume(self, plan_id: str, reason: str = "already used (plan handles are single-use)") -> StoredPlan:
        with self._lock:
            stored = self._get_locked(plan_id)
            self._plans.pop(plan_id, None)
            self._gone[plan_id] = reason
            return stored

    def invalidate_all(self, reason: str) -> int:
        with self._lock:
            n = len(self._plans)
            for pid in list(self._plans):
                self._gone[pid] = f"invalidated: {reason}"
            self._plans.clear()
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._plans)

    def _get_locked(self, plan_id: str) -> StoredPlan:
        if plan_id in self._gone:
            reason = self._gone[plan_id]
            exc = PlanAlreadyUsed if reason.startswith("already used") else PlanInvalidated
            if reason == "expired":
                exc = PlanExpired
            raise exc(f"plan {plan_id} cannot be used: {reason}. Plan again.")
        stored = self._plans.get(plan_id)
        if stored is None:
            raise PlanNotFound(f"unknown plan id {plan_id!r}. Plan again with plan_to_joints / plan_to_pose.")
        if self._clock() >= stored.expires_at:
            self._plans.pop(plan_id, None)
            self._gone[plan_id] = "expired"
            raise PlanExpired(f"plan {plan_id} cannot be used: expired (ttl {self.ttl_s:g} s). Plan again.")
        return stored

    def _purge_locked(self) -> None:
        now = self._clock()
        for pid, sp in list(self._plans.items()):
            if now >= sp.expires_at:
                self._plans.pop(pid)
                self._gone[pid] = "expired"
        if len(self._gone) > 4096:  # bound memory
            for pid in list(self._gone)[:2048]:
                self._gone.pop(pid)
