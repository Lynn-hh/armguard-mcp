"""Minimal MCP client for armguard-mcp over stdio (MCP Python SDK v2).

Launches the server as a subprocess with the fake FR3 backend in dry-run mode, lists the
tools, plans a small motion and "executes" it (dry run: nothing moves). Approval prompts, if
any, are answered by ``ask_human`` on the terminal.

    python examples/python_client.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anyio
import mcp.types as mt
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

POLICY = Path(__file__).resolve().parent / "policies" / "fr3.yaml"


async def ask_human(context, params: mt.ElicitRequestParams) -> mt.ElicitResult:
    print(f"\n{params.message}\n", file=sys.stderr)
    answer = (await anyio.to_thread.run_sync(input, "approve? [y/N] ")).strip().lower()
    if answer != "y":
        return mt.ElicitResult(action="decline")
    return mt.ElicitResult(action="accept", content={"approve": True, "operator": "cli-user"})


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "armguard_mcp", "--policy", str(POLICY), "--backend", "fake", "--dry-run"],
    )
    async with Client(server, elicitation_callback=ask_human) as client:
        tools = await client.list_tools()
        print("tools:", ", ".join(t.name for t in tools.tools))

        plan = await client.call_tool("plan_to_pose", {"position": {"x": 0.40, "y": 0.0, "z": 0.40}})
        summary = plan.structured_content
        print("plan:", summary["plan_id"], summary["status"], summary["violations"])

        result = await client.call_tool("execute_plan", {"plan_id": summary["plan_id"]})
        print("execute:", result.structured_content or result.content[0].text)


if __name__ == "__main__":
    anyio.run(main)
