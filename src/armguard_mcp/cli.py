"""Command-line entry point: ``armguard-mcp --policy fr3.yaml [--backend fake|ros2] [--transport stdio|http]``.

On the stdio transport stdout carries the MCP protocol, so all logging goes to stderr.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from armguard_mcp import __version__

logger = logging.getLogger("armguard_mcp")

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="armguard-mcp",
        description="Safety-first MCP server for ROS 2 manipulators (server-side safety envelope).",
    )
    p.add_argument("--policy", required=True, help="path to the safety policy YAML")
    p.add_argument(
        "--backend", choices=["fake", "ros2"], default="fake", help="robot backend (default: fake)"
    )
    p.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio", help="MCP transport (default: stdio)"
    )
    p.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    p.add_argument("--audit-log", default=None, help="append-only JSONL audit log path")
    p.add_argument("--dry-run", action="store_true", help="force dry-run on top of the policy: nothing moves")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr, level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Imports deferred so `--help` stays fast and import errors are reported cleanly.
    import anyio

    from armguard_mcp.backends import create_backend
    from armguard_mcp.policy import PolicyError, load_policy
    from armguard_mcp.safety.audit import AuditLogger
    from armguard_mcp.server import build

    try:
        policy = load_policy(args.policy)
    except PolicyError as e:
        print(f"armguard-mcp: {e}", file=sys.stderr)
        return 2
    if args.dry_run:
        policy = policy.with_dry_run(True)

    try:
        backend = create_backend(args.backend, policy)
    except (RuntimeError, ValueError) as e:
        print(f"armguard-mcp: {e}", file=sys.stderr)
        return 2

    audit = AuditLogger(args.audit_log)
    app = build(policy, backend, audit)
    logger.info(
        "armguard-mcp %s: robot=%s backend=%s transport=%s dry_run=%s groups=%s",
        __version__,
        policy.robot.name,
        args.backend,
        args.transport,
        policy.dry_run,
        ",".join(policy.tools.enabled),
    )
    try:
        if args.transport == "stdio":
            app.server.run("stdio")
        else:
            if args.host not in _LOOPBACK:
                logger.warning(
                    "binding to %s: the HTTP transport has NO authentication in this build; anyone who can reach "
                    "this port can drive the robot (within the policy). Prefer 127.0.0.1 plus an SSH tunnel.",
                    args.host,
                )
            app.server.run("streamable-http", host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            anyio.run(backend.shutdown)
        except Exception:  # pragma: no cover - best effort
            logger.exception("backend shutdown failed")
        audit.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
