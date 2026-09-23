# Contributing to armguard-mcp

Thanks for your interest. This project controls physical robots, so the bar is **correctness, and honesty
about behaviour**, before features. A small PR with a test beats a large one without.

## Development setup

No ROS is needed for the core. Development happens on Python 3.12, and CI also runs 3.10.

```bash
git clone https://github.com/Lynn-hh/armguard-mcp && cd armguard-mcp
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # add ,image to exercise Pillow downscaling
```

## Tests

```bash
pytest -q tests                  # unit + in-memory MCP + stdio subprocess tests (fake FR3)
python scripts/demo_fake.py      # scripted end-to-end session
```

- Async tests use anyio's pytest plugin (`@pytest.mark.anyio`, with the `anyio_backend` fixture in
  `tests/conftest.py` pinned to asyncio). No `pytest-asyncio` is needed.
- Server tests talk to the real server through `mcp.client.Client(server, mode=...)`. **Any change to an
  approval flow must be tested in `auto`, `legacy` and `2026-07-28` modes**, because under 2026-07-28
  resolvers run more than once per call.
- Use `FakeClock` from `tests/conftest.py` for TTLs and rate limits instead of sleeping.
- ROS 2 backend tests live in `tests_ros/`. They need a sourced ROS 2 Jazzy environment and run in the
  `ros-jazzy` CI job (container `ros:jazzy`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`). Locally:

  ```bash
  source /opt/ros/jazzy/setup.bash
  python3 -m venv --system-site-packages .venv-ros && . .venv-ros/bin/activate
  pip install -e ".[dev]" && pytest -q tests_ros
  ```

## Style

```bash
ruff check . && ruff format --check .
```

- Line length is 110 (`ruff format` enforces it). Lint rules are in `pyproject.toml`.
- Type-annotate everything, and prefer `async def` for tools and backend methods.
- **Do not add `from __future__ import annotations` to `server.py`.** The SDK needs real annotations
  there to resolve `Resolve(...)` parameters, and without them the hidden `approval` parameter would leak
  into the input schema.
- **Approval resolvers must stay pure:** no audit writes, no plan consumption, no rate-limit
  consumption, no motion. Put side effects in the tool body, and re-check there.
- Nothing may be written to stdout (the stdio transport). Use `logging`, which goes to stderr.
- Error messages go to the model, so make them say what was refused and what to do instead.
- Keep `rclpy` imports inside `armguard_mcp/backends/ros2.py`.

## Changes that touch safety

If your PR changes the envelope, approvals, rate limits, the e-stop, plan handling or the audit log:

1. Add or extend a test that fails without your change.
2. Update [docs/threat-model.md](docs/threat-model.md) and the README's policy reference or approval
   section when behaviour changes.
3. Regenerate the tool table if tools or annotations change: `python scripts/gen_tool_table.py`.
4. Say in the PR description what could go wrong on a real robot.

Never weaken a default (`approval.on_client_without_elicitation: deny`, loopback-only HTTP, the always-on
safety group) without discussing it in an issue first.

## Reporting security issues

Please do not open a public issue for vulnerabilities that could let an LLM or client bypass the safety
envelope or approvals. Email the maintainer (see `pyproject.toml`) instead.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
