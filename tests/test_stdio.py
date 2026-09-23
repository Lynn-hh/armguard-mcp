"""stdio hygiene: nothing but MCP protocol may appear on stdout, and the CLI works as a real subprocess."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from tests.conftest import FR3_POLICY, ROOT

ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def test_import_and_build_print_nothing_to_stdout() -> None:
    code = (
        "import armguard_mcp.server as s, armguard_mcp.cli, armguard_mcp.backends.fake as f;"
        "from armguard_mcp.policy import load_policy;"
        f"p = load_policy({str(FR3_POLICY)!r});"
        "s.build_server(p, f.FakeBackend.from_policy(p))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=ENV, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout == ""


def test_package_does_not_import_rclpy() -> None:
    code = "import sys, armguard_mcp.server, armguard_mcp.backends; print('rclpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=ENV, timeout=60)
    assert out.stdout.strip() == "False"


def test_cli_rejects_bad_policy(tmp_path) -> None:
    bad = tmp_path / "p.yaml"
    bad.write_text("version: 1\nrobot: {}\n")
    out = subprocess.run(
        [sys.executable, "-m", "armguard_mcp", "--policy", str(bad)], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 2 and "invalid policy" in out.stderr and out.stdout == ""


def test_cli_ros2_backend_unavailable_without_ros() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "armguard_mcp", "--policy", str(FR3_POLICY), "--backend", "ros2"],
        capture_output=True,
        text=True,
        env=ENV,
    )
    assert out.returncode == 2 and "ros2 backend is unavailable" in out.stderr


@pytest.mark.anyio
async def test_stdio_end_to_end(tmp_path) -> None:
    audit = tmp_path / "audit.jsonl"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "armguard_mcp", "--policy", str(FR3_POLICY), "--audit-log", str(audit), "--dry-run"],
        env=ENV,
    )
    async with Client(params) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"execute_plan", "estop", "stop_motion"} <= names
        st = await c.call_tool("get_safety_status", {})
        assert not st.is_error and st.structured_content["dry_run"] is True
    assert '"tool":"get_safety_status"' in audit.read_text()
