"""Robot backends. ``rclpy`` is only ever imported by the ros2 backend, lazily."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from armguard_mcp.backends.base import (
    BackendError,
    BackendFailed,
    BackendTimeout,
    CameraFrame,
    NotSupported,
    RobotBackend,
)

if TYPE_CHECKING:
    from armguard_mcp.policy import Policy

__all__ = [
    "BackendError",
    "BackendFailed",
    "BackendTimeout",
    "CameraFrame",
    "NotSupported",
    "RobotBackend",
    "create_backend",
]


def create_backend(name: str, policy: Policy) -> RobotBackend:
    """Instantiate a backend by name (``fake`` or ``ros2``)."""
    if name == "fake":
        from armguard_mcp.backends.fake import FakeBackend

        return FakeBackend.from_policy(policy)
    if name == "ros2":
        try:
            module = importlib.import_module("armguard_mcp.backends.ros2")
        except ImportError as e:
            raise RuntimeError(
                "the ros2 backend is unavailable: it needs a sourced ROS 2 environment (rclpy, "
                f"MoveIt 2, franka_ros2) and the armguard_mcp.backends.ros2 module ({e})"
            ) from e
        return module.Ros2Backend.from_policy(policy)
    raise ValueError(f"unknown backend {name!r} (expected 'fake' or 'ros2')")
