"""Append-only JSONL audit log plus an in-memory ring buffer for ``get_audit_tail``.

Every tool call is recorded (successes, denials and errors), as are approval decisions,
e-stops and force-limit violations. Binary payloads and image data are redacted.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REDACT_KEYS = frozenset({"data", "image", "image_data", "png", "jpeg", "bytes", "blob"})
_MAX_STR = 2048


def iso_utc(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def redact(value: Any, key: str | None = None) -> Any:
    """Recursively make ``value`` JSON-safe and strip image/binary payloads."""
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<redacted {len(value)} bytes>"
    if key is not None and key.lower() in _REDACT_KEYS and isinstance(value, str):
        return f"<redacted {len(value)} chars>"
    if isinstance(value, str):
        return (
            value if len(value) <= _MAX_STR else value[:_MAX_STR] + f"...<truncated {len(value) - _MAX_STR}>"
        )
    if isinstance(value, Mapping):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [redact(v) for v in value]
    if hasattr(value, "model_dump"):
        return redact(value.model_dump(mode="json"), key)
    if value is None or isinstance(value, bool | int | float):
        return value
    return repr(value)


class AuditLogger:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        ring_size: int = 500,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock
        self._ring: deque[dict[str, Any]] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._seq = 0
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Append-only; line-buffered so each event is flushed as one line.
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def log(
        self,
        event: str,
        *,
        tool: str | None = None,
        args: Mapping[str, Any] | None = None,
        outcome: str | None = None,
        verdict: Any = None,
        approval: Mapping[str, Any] | None = None,
        plan_id: str | None = None,
        error: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            record: dict[str, Any] = {"seq": self._seq, "ts": iso_utc(self._clock()), "event": event}
            fields = {
                "tool": tool,
                "args": args,
                "outcome": outcome,
                "plan_id": plan_id,
                "verdict": verdict,
                "approval": approval,
                "error": error,
                **extra,
            }
            record.update({k: redact(v, k) for k, v in fields.items() if v is not None})
            self._ring.append(record)
            if self._fh is not None:
                try:
                    self._fh.write(json.dumps(record, separators=(",", ":"), sort_keys=False) + "\n")
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                except OSError:  # never let audit I/O crash a stop/estop path
                    logger.exception("failed to write audit record")
            return record

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._ring)[-n:] if n > 0 else []

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
