"""Always-collected sanity checks, so ``pytest tests_ros`` without ROS reports skips (exit 0)."""

from __future__ import annotations

import os

import pytest

from tests_ros.conftest import HAVE_RCLPY


@pytest.mark.skipif(not HAVE_RCLPY, reason="needs a sourced ROS 2 environment (rclpy)")
def test_tests_are_isolated_from_the_network() -> None:
    assert os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"
    assert 0 < int(os.environ["ROS_DOMAIN_ID"]) <= 101


@pytest.mark.skipif(not HAVE_RCLPY, reason="needs a sourced ROS 2 environment (rclpy)")
def test_required_interface_packages_are_installed() -> None:
    import importlib

    for pkg in ("moveit_msgs.srv", "control_msgs.action", "controller_manager_msgs.srv", "tf2_ros"):
        importlib.import_module(pkg)
