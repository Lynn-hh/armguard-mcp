"""Token-bucket rate limiting per tool plus a global bucket.

Safety tools (stop_motion, estop, get_safety_status) are never rate limited: an agent (or a
human driving it) must always be able to stop the robot.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from armguard_mcp.policy import RateLimitsSection

NEVER_LIMITED: frozenset[str] = frozenset({"stop_motion", "estop", "get_safety_status"})


class RateLimitExceeded(Exception):
    def __init__(self, tool: str, scope: str, retry_after_s: float) -> None:
        self.tool, self.scope, self.retry_after_s = tool, scope, retry_after_s
        super().__init__(
            f"rate limit exceeded for {tool} ({scope} bucket); retry in {max(retry_after_s, 0.0):.1f} s"
        )


@dataclass
class TokenBucket:
    capacity: float
    refill_per_s: float
    tokens: float
    updated: float

    @classmethod
    def per_minute(cls, n: int, now: float) -> TokenBucket:
        return cls(capacity=float(n), refill_per_s=n / 60.0, tokens=float(n), updated=now)

    def _refill(self, now: float) -> None:
        if now > self.updated:
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_s)
            self.updated = now

    def available(self, now: float) -> bool:
        self._refill(now)
        return self.tokens >= 1.0

    def retry_after(self, now: float) -> float:
        self._refill(now)
        return 0.0 if self.tokens >= 1.0 else (1.0 - self.tokens) / self.refill_per_s

    def take(self) -> None:
        self.tokens -= 1.0


class RateLimiter:
    def __init__(self, cfg: RateLimitsSection, clock: Callable[[], float] = time.monotonic) -> None:
        self._cfg = cfg
        self._clock = clock
        self._lock = threading.Lock()
        self._global = TokenBucket.per_minute(cfg.global_per_minute, clock())
        self._tools: dict[str, TokenBucket] = {}

    def _bucket(self, tool: str, now: float) -> TokenBucket:
        b = self._tools.get(tool)
        if b is None:
            b = TokenBucket.per_minute(self._cfg.per_tool.get(tool, self._cfg.default_per_minute), now)
            self._tools[tool] = b
        return b

    def would_allow(self, tool: str) -> bool:
        """Non-consuming check (safe to call from pure approval resolvers)."""
        if tool in NEVER_LIMITED:
            return True
        with self._lock:
            now = self._clock()
            return self._global.available(now) and self._bucket(tool, now).available(now)

    def acquire(self, tool: str) -> None:
        """Consume one token from both the tool and global buckets, atomically, or raise."""
        if tool in NEVER_LIMITED:
            return
        with self._lock:
            now = self._clock()
            bucket = self._bucket(tool, now)
            if not bucket.available(now):
                raise RateLimitExceeded(tool, "per-tool", bucket.retry_after(now))
            if not self._global.available(now):
                raise RateLimitExceeded(tool, "global", self._global.retry_after(now))
            bucket.take()
            self._global.take()
