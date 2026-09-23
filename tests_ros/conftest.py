"""Fixtures for ROS 2 integration tests (run inside a sourced ROS 2 Jazzy environment).

Every test module starts with ``pytest.importorskip("rclpy")``, so ``pytest tests_ros`` on a
machine without ROS reports skipped modules instead of errors.

Isolation: a random ``ROS_DOMAIN_ID`` and ``ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`` are set
before any rclpy context is initialised, so tests never talk to real robots on the network.
"""

from __future__ import annotations

import importlib.util
import os
import random
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

HAVE_RCLPY = importlib.util.find_spec("rclpy") is not None

if HAVE_RCLPY:
    os.environ["ROS_DOMAIN_ID"] = os.environ.get("ARMGUARD_TEST_DOMAIN_ID", str(random.randint(1, 100)))
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.pop("ROS_LOCALHOST_ONLY", None)  # deprecated on Jazzy; would conflict


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"  # the ros2 backend bridges rclpy futures with asyncio.call_soon_threadsafe


@pytest.fixture
def robot() -> Iterator[Any]:
    from tests_ros.fake_robot import FakeRosRobot

    r = FakeRosRobot(speedup=1.0)
    try:
        yield r
    finally:
        r.close()


def ros2_section(**over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {"startup_timeout_s": 10.0, "service_timeout_s": 5.0, "fk_source": "moveit"}
    cfg.update(over)
    return cfg


def ros_policy(ros2: dict[str, Any] | None = None, **over: Any) -> Any:
    from tests.conftest import make_policy

    return make_policy(ros2=ros2_section(**(ros2 or {})), **over)


@asynccontextmanager
async def running_backend(policy: Any) -> AsyncIterator[Any]:
    from armguard_mcp.backends.ros2 import Ros2Backend

    backend = Ros2Backend.from_policy(policy)
    await backend.start()
    try:
        yield backend
    finally:
        await backend.shutdown()
