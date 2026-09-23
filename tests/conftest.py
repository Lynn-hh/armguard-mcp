from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import mcp.types as mt
import pytest
import yaml

from armguard_mcp.backends.fake import FakeBackend
from armguard_mcp.policy import Policy
from armguard_mcp.safety.audit import AuditLogger
from armguard_mcp.server import ArmGuardApp, build

ROOT = Path(__file__).resolve().parents[1]
FR3_POLICY = ROOT / "examples" / "policies" / "fr3.yaml"
READONLY_POLICY = ROOT / "examples" / "policies" / "readonly.yaml"
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def fr3_dict() -> dict[str, Any]:
    return yaml.safe_load(FR3_POLICY.read_text())


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_policy(**overrides: Any) -> Policy:
    return Policy.from_dict(deep_merge(fr3_dict(), overrides))


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_app(
    policy: Policy | None = None,
    *,
    audit_path: Path | None = None,
    clock: Any = None,
    monotonic: Any = None,
    **backend_kwargs: Any,
) -> ArmGuardApp:
    policy = policy or make_policy()
    backend_kwargs.setdefault("speedup", 100.0)
    backend = FakeBackend.from_policy(policy, **backend_kwargs)
    kw: dict[str, Any] = {}
    if clock is not None:
        kw["clock"] = clock
    if monotonic is not None:
        kw["monotonic"] = monotonic
    audit = AuditLogger(audit_path, clock=clock) if clock is not None else AuditLogger(audit_path)
    return build(policy, backend, audit, **kw)


class Elicitor:
    """Scripted human: records every approval prompt and answers with a fixed action."""

    def __init__(self, action: str = "accept", approve: bool = True, operator: str = "lynn") -> None:
        self.action, self.approve, self.operator = action, approve, operator
        self.messages: list[str] = []

    async def __call__(self, context: Any, params: Any) -> mt.ElicitResult:
        self.messages.append(params.message)
        if self.action == "accept":
            return mt.ElicitResult(
                action="accept", content={"approve": self.approve, "operator": self.operator}
            )
        return mt.ElicitResult(action=self.action)  # type: ignore[arg-type]


def text(result: Any) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


def audit_events(app: ArmGuardApp, tool: str | None = None) -> list[dict[str, Any]]:
    events = app.guard.audit.tail(500)
    return [e for e in events if tool is None or e.get("tool") == tool]
