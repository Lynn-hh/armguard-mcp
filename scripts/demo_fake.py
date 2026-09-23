"""Scripted armguard-mcp session against the simulated FR3 (no ROS needed).

Plays the role of an MCP client (the LLM side) plus a human who approves prompts, talking to
the real server over an in-memory MCP session. The output is what docs/demo.md shows.

    python scripts/demo_fake.py [--mode auto|legacy|2026-07-28] [--full]

Sequence:
  1. get_robot_state
  2. plan_to_pose (small move, inside the envelope) -> execute_plan (no prompt needed)
  3. plan_to_joints (large joint-1 rotation: soft warning) -> execute_plan -> human approves
     (the prompt is shown on stdout; the scripted human always approves)
  4. plan_to_pose into the 'camera_mount' keep-out zone -> rejected; execute_plan refused
  5. estop -> planning is refused while e-stopped
  6. get_audit_tail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anyio
import mcp.types as mt
from mcp.client import Client

from armguard_mcp.backends.fake import FakeBackend
from armguard_mcp.policy import load_policy
from armguard_mcp.safety.audit import AuditLogger
from armguard_mcp.server import build

ROOT = Path(__file__).resolve().parents[1]
FULL = False


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def show(value: Any, keys: list[str] | None = None) -> str:
    if keys is not None and not FULL and isinstance(value, dict):
        value = {k: value[k] for k in keys if k in value}
    # Floats are rounded to 4 decimals for readability (use --full for the raw result).
    return json.dumps(value if FULL else rounded(value), indent=2 if FULL else None)


async def call(client: Client, name: str, args: dict[str, Any] | None = None, keys: list[str] | None = None):
    print(f"\n>>> {name}({json.dumps(args or {})})")
    progress: list[float] = []

    async def on_progress(p: float, total: float | None, message: str | None) -> None:
        progress.append(p)

    r = await client.call_tool(name, args or {}, progress_callback=on_progress)
    if r.is_error:
        print("ERROR: " + " ".join(getattr(c, "text", "") for c in r.content))
    elif r.structured_content is not None:
        text = show(r.structured_content, keys)
        print(text if FULL else _wrap(text))
    if progress:
        print(f"(progress notifications received: {len(progress)}, last={progress[-1]:.2f})")
    return r


def _wrap(text: str) -> str:
    """Put each top-level key on its own line so compact JSON stays readable."""
    if not text.startswith("{"):
        return text
    obj = json.loads(text)
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in obj.items())
    return "{\n" + body + "\n}"


class Human:
    """Approves every prompt and prints what the operator was shown."""

    async def __call__(self, context: Any, params: Any) -> mt.ElicitResult:
        print("\n--- approval prompt shown to the human (MCP elicitation) ---")
        print(params.message)
        print("--- human ticks 'approve', operator='lynn' ---")
        return mt.ElicitResult(action="accept", content={"approve": True, "operator": "lynn"})


PLAN_KEYS = [
    "plan_id",
    "kind",
    "status",
    "executable",
    "duration_s",
    "max_joint_travel_rad",
    "max_joint_velocity_ratio",
    "violations",
    "requires_approval",
]
EXEC_KEYS = ["plan_id", "status", "message", "executed", "approval", "max_observed_force_n"]


async def main(mode: str, policy_path: Path) -> None:
    policy = load_policy(policy_path)
    app = build(policy, FakeBackend.from_policy(policy), AuditLogger())
    async with Client(app.server, mode=mode, elicitation_callback=Human()) as c:
        print(f"# armguard-mcp demo: fake FR3, policy {policy_path.name}, protocol mode {mode!r}")

        print("\n## 1. Look before moving")
        r = await call(c, "get_robot_state", keys=["robot", "backend", "ee_pose", "wrench", "safety"])

        print("\n## 2. Small move inside the envelope: no approval needed (approval.mode=outside_envelope)")
        r = await call(c, "plan_to_pose", {"position": {"x": 0.40, "y": 0.10, "z": 0.40}}, PLAN_KEYS)
        await call(c, "execute_plan", {"plan_id": r.structured_content["plan_id"]}, EXEC_KEYS)
        print("\n# Plan handles are single-use:")
        await call(c, "execute_plan", {"plan_id": r.structured_content["plan_id"]})

        print("\n## 3. Large motion (joint 1 turns -0.9 rad > soft_joint_step_rad 0.8): a human must approve")
        js = (await c.call_tool("get_robot_state", {})).structured_content["joint_state"]["positions"]
        target = list(js)
        target[0] -= 0.9
        r = await call(c, "plan_to_joints", {"joint_positions": [round(v, 4) for v in target]}, PLAN_KEYS)
        await call(c, "execute_plan", {"plan_id": r.structured_content["plan_id"]}, EXEC_KEYS)

        print("\n## 4. Unsafe request: TCP into the 'camera_mount' keep-out zone")
        r = await call(c, "plan_to_pose", {"position": {"x": 0.62, "y": 0.42, "z": 0.30}}, PLAN_KEYS)
        print("\n# Even if the model tries anyway, a rejected plan can never be executed:")
        await call(c, "execute_plan", {"plan_id": r.structured_content["plan_id"]})

        print("\n## 5. Software e-stop latches; motion tools are refused until a human resets it")
        await call(c, "estop", {"reason": "demo: operator saw something odd"})
        await call(c, "plan_to_pose", {"position": {"x": 0.40, "y": 0.0, "z": 0.40}})

        print("\n## 6. Audit trail (get_audit_tail; last 9 events before this call, abridged)")
        tail = (await c.call_tool("get_audit_tail", {"n": 10})).structured_content["events"][:-1]
        for e in tail:
            brief = {k: e[k] for k in ("seq", "event", "tool", "outcome", "plan_id", "approval") if k in e}
            print(json.dumps(brief))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scripted armguard-mcp session on the fake FR3 backend.")
    ap.add_argument("--mode", default="auto", choices=["auto", "legacy", "2026-07-28"])
    ap.add_argument("--policy", type=Path, default=ROOT / "examples" / "policies" / "fr3.yaml")
    ap.add_argument("--full", action="store_true", help="print complete tool results")
    a = ap.parse_args()
    FULL = a.full
    anyio.run(main, a.mode, a.policy)
