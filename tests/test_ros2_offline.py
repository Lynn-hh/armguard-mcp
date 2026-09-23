"""ROS-free parts of the ros2 backend: configuration, lazy imports and message conversion helpers.

The backend itself is exercised against real rclpy endpoints in ``tests_ros/`` (ROS 2 Jazzy).
"""

from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from types import SimpleNamespace as NS

import pytest

from armguard_mcp.backends.base import BackendFailed, NotSupported
from armguard_mcp.backends.ros2 import (
    compressed_image_info,
    filter_graph,
    raw_image_to_png,
    s_to_sec_nanosec,
    trajectory_arrays,
)
from armguard_mcp.backends.ros2_config import Ros2BackendConfig
from armguard_mcp.imaging import gradient_png, pillow_available
from armguard_mcp.policy import PolicyError, load_ros2_config
from tests.conftest import FR3_POLICY, make_policy

J = [f"j{i}" for i in range(3)]


def dur(t: float) -> NS:
    sec, nsec = s_to_sec_nanosec(t)
    return NS(sec=sec, nanosec=nsec)


def point(q, t, v=(), a=()) -> NS:
    return NS(positions=list(q), velocities=list(v), accelerations=list(a), time_from_start=dur(t))


# --- lazy import / CLI ----------------------------------------------------------------------
def test_importing_the_backend_does_not_import_rclpy() -> None:
    code = "import sys, armguard_mcp.backends.ros2, armguard_mcp.policy; print('rclpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_create_backend_without_rclpy_explains_what_to_do() -> None:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("rclpy is importable here")
    from armguard_mcp.backends import create_backend

    with pytest.raises(RuntimeError, match=r"source /opt/ros/jazzy/setup\.bash"):
        create_backend("ros2", make_policy())


def test_cli_rejects_a_bad_ros2_config(tmp_path) -> None:
    bad = tmp_path / "ros2.yaml"
    bad.write_text("ros2:\n  trajectory_actoin: /x\n")
    out = subprocess.run(
        [sys.executable, "-m", "armguard_mcp", "--policy", str(FR3_POLICY), "--ros2-config", str(bad)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 2 and "trajectory_actoin" in out.stderr


# --- configuration --------------------------------------------------------------------------
def test_config_defaults_follow_franka_ros2() -> None:
    c = Ros2BackendConfig()
    assert c.trajectory_action == "/fr3_arm_controller/follow_joint_trajectory"
    assert c.gripper_namespace == "/franka_gripper"
    assert c.collision_behavior_service == "/service_server/set_force_torque_collision_behavior"
    assert c.error_recovery_action == "/action_server/error_recovery"
    assert c.require_wrench and c.fk_source == "moveit"


def test_policy_ros2_section_is_strict() -> None:
    p = make_policy(ros2={"wrench_topic": None, "planning_time_s": 2.0})
    assert p.ros2 is not None and p.ros2.wrench_topic is None and p.ros2.planning_time_s == 2.0
    assert make_policy().ros2 is None
    with pytest.raises(PolicyError, match=r"ros2\.no_such_field"):
        make_policy(ros2={"no_such_field": 1})
    with pytest.raises(PolicyError, match="7 positive values"):
        make_policy(ros2={"collision_joint_torque_thresholds_nm": [1, 2, 3]})


def test_example_ros2_config_loads() -> None:
    from tests.conftest import ROOT

    cfg = load_ros2_config(ROOT / "examples" / "ros2" / "fr3_franka_ros2_jazzy.yaml")
    assert cfg == Ros2BackendConfig()  # the example documents the defaults explicitly


def test_load_ros2_config_file(tmp_path) -> None:
    f = tmp_path / "a.yaml"
    f.write_text("ros2:\n  joint_states_topic: /robot/joint_states\n")
    assert load_ros2_config(f).joint_states_topic == "/robot/joint_states"
    f.write_text("joint_states_topic: /other\n")
    assert load_ros2_config(f).joint_states_topic == "/other"
    f.write_text("- not a mapping\n")
    with pytest.raises(PolicyError):
        load_ros2_config(f)


# --- trajectory conversion ------------------------------------------------------------------
def test_trajectory_is_reordered_to_policy_joint_order() -> None:
    pts = [point([3, 2, 1], 0.0, [0, 0, 0]), point([6, 5, 4], 1.5, [0.3, 0.2, 0.1])]
    arr = trajectory_arrays(["j2", "j1", "j0"], pts, J)
    assert arr.waypoints == [[1, 2, 3], [4, 5, 6]]
    assert arr.times == [0.0, 1.5]
    assert arr.velocities == [[0, 0, 0], [0.1, 0.2, 0.3]]
    assert arr.accelerations is None  # not given for every point


def test_trajectory_rejections() -> None:
    ok = [point([0, 0, 0], 0.0), point([1, 1, 1], 1.0)]
    with pytest.raises(BackendFailed, match="lacks joints"):
        trajectory_arrays(["j0", "j1"], ok, J)
    with pytest.raises(BackendFailed, match="does not cover"):
        trajectory_arrays([*J, "finger"], [point([0, 0, 0, 0], 0.0), point([1, 1, 1, 1], 1.0)], J)
    with pytest.raises(BackendFailed, match="empty"):
        trajectory_arrays(J, [], J)
    with pytest.raises(BackendFailed, match="not time-parameterised"):
        trajectory_arrays(J, [point([0, 0, 0], 0.0), point([1, 1, 1], 0.0)], J)
    with pytest.raises(BackendFailed, match="decreasing"):
        trajectory_arrays(J, [point([0, 0, 0], 0.0), point([1, 1, 1], 1.0), point([2, 2, 2], 0.5)], J)
    with pytest.raises(BackendFailed, match="non-finite"):
        trajectory_arrays(J, [point([0, 0, 0], 0.0), point([float("nan"), 1, 1], 1.0)], J)


def test_single_point_trajectory_becomes_a_short_hold() -> None:
    arr = trajectory_arrays(J, [point([1, 2, 3], 0.0, [0, 0, 0])], J)
    assert arr.waypoints == [[1, 2, 3], [1, 2, 3]] and arr.times == [0.0, 0.1]
    assert arr.velocities == [[0.0] * 3, [0.0] * 3]


def test_duration_rounding() -> None:
    assert s_to_sec_nanosec(1.25) == (1, 250_000_000)
    assert s_to_sec_nanosec(0.9999999999) == (1, 0)


# --- images ---------------------------------------------------------------------------------
def decode_png_rgb(data: bytes) -> tuple[int, int, list[bytes]]:
    """Minimal decoder for the unfiltered RGB PNGs written by armguard_mcp.imaging."""
    w, h = struct.unpack(">II", data[16:24])
    i, idat = 8, b""
    while i < len(data):
        (n,) = struct.unpack(">I", data[i : i + 4])
        kind = data[i + 4 : i + 8]
        if kind == b"IDAT":
            idat += data[i + 8 : i + 8 + n]
        i += 12 + n
    raw = zlib.decompress(idat)
    stride = 1 + 3 * w
    rows = [raw[r * stride : (r + 1) * stride] for r in range(h)]
    assert all(r[0] == 0 for r in rows)
    return w, h, [r[1:] for r in rows]


@pytest.mark.skipif(pillow_available(), reason="checks the pure-Python path")
def test_raw_bgr_image_is_converted_and_decimated() -> None:
    w, h = 4, 2
    # pixel (x, y) = BGR (x, y, 200)
    data = bytes(v for y in range(h) for x in range(w) for v in (x, y, 200))
    png, nw, nh = raw_image_to_png("bgr8", w, h, 3 * w, data, max_width=2)
    assert (nw, nh) == (2, 1)
    dw, dh, rows = decode_png_rgb(png)
    assert (dw, dh) == (2, 1)
    assert rows[0] == bytes([200, 0, 0, 200, 0, 2])  # RGB of x=0 and x=2 in row 0


@pytest.mark.skipif(pillow_available(), reason="checks the pure-Python path")
def test_raw_mono_and_row_padding() -> None:
    w, h, step = 3, 2, 4  # one padding byte per row
    data = bytes([10, 20, 30, 99, 40, 50, 60, 99])
    png, nw, nh = raw_image_to_png("mono8", w, h, step, data, max_width=640)
    assert (nw, nh) == (3, 2)
    _, _, rows = decode_png_rgb(png)
    assert rows[1] == bytes([40, 40, 40, 50, 50, 50, 60, 60, 60])


def test_raw_image_errors() -> None:
    with pytest.raises(NotSupported, match="16UC1"):
        raw_image_to_png("16UC1", 2, 2, 4, bytes(8), 640)
    with pytest.raises(BackendFailed, match="malformed"):
        raw_image_to_png("rgb8", 4, 4, 12, bytes(10), 640)


def test_compressed_image_info() -> None:
    assert compressed_image_info(gradient_png(64, 48)) == ("image/png", 64, 48)
    # SOI, APP0 (len 16), SOF0 (len 17): 8-bit, height 480, width 640
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + bytes(9)
    sof0 = b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 480, 640) + bytes(10)
    assert compressed_image_info(b"\xff\xd8" + app0 + sof0 + b"\xff\xd9") == ("image/jpeg", 640, 480)
    with pytest.raises(NotSupported):
        compressed_image_info(b"GIF89a....")


def test_graph_filtering() -> None:
    g = filter_graph(
        nodes=["/armguard_mcp", "/move_group", "/move_group"],
        topics=["/joint_states", "/x/_action/feedback"],
        services=["/compute_fk", "/move_group/get_parameters", "/x/_action/send_goal"],
        actions=["/x"],
        own_node="/armguard_mcp",
    )
    assert g.nodes == ["/move_group"] and g.topics == ["/joint_states"]
    assert g.services == ["/compute_fk"] and g.actions == ["/x"]
