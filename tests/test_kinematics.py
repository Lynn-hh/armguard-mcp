from __future__ import annotations

import math
import random

import pytest

from armguard_mcp import geometry as g
from armguard_mcp.kinematics import FR3_JOINT_LIMITS, FR3_READY, FR3Kinematics


def planar_ready_tcp() -> tuple[float, float]:
    """Independent derivation of the TCP at the ready pose (q1 = q3 = q5 = 0, so the arm is planar in x-z).

    * Shoulder (joint 2 axis) at z = 0.333.
    * Upper arm: 0.316 m tilted by q2 = -pi/4 from vertical (towards -x), then the 0.0825 m
      elbow offset perpendicular to it (towards +x, +z).
    * q4 = -3pi/4 makes the forearm horizontal (pi/2 from vertical): 0.384 m along +x, with the
      -0.0825 m offset now pointing straight up (+z).
    * q6 = pi/2 folds the wrist so the flange points straight down; the 0.088 m link-7 offset is along +x.
    * Flange (0.107 m) + Franka Hand TCP (0.1034 m) hang straight down.
    """
    s = math.sin(math.pi / 4)
    x = -0.316 * s + 0.0825 * s + 0.384 + 0.088
    z = 0.333 + 0.316 * s + 0.0825 * s + 0.0825 - (0.107 + 0.1034)
    return x, z


def test_fk_ready_pose_matches_planar_derivation() -> None:
    k = FR3Kinematics()
    t = k.fk(FR3_READY)
    x, z = planar_ready_tcp()
    px, py, pz = g.translation(t)
    assert px == pytest.approx(x, abs=1e-6)
    assert py == pytest.approx(0.0, abs=1e-9)
    assert pz == pytest.approx(z, abs=1e-6)
    # Also matches the widely quoted libfranka O_T_EE for the ready pose (0.3069, 0, 0.4869).
    assert (px, pz) == pytest.approx((0.3069, 0.4869), abs=5e-4)
    # Hand points straight down: TCP z-axis = -base z; x-axis along +base x.
    r = g.rotation(t)
    assert [r[0][2], r[1][2], r[2][2]] == pytest.approx([0, 0, -1], abs=1e-9)
    assert [r[0][0], r[1][0], r[2][0]] == pytest.approx([1, 0, 0], abs=1e-9)


def test_fk_shoulder_geometry() -> None:
    k = FR3Kinematics()
    q = [0.0, 0.0, 0.0, -0.1518, 0.0, 0.5445, 0.0]
    frames = k.chain(q)
    # joint 1 and 2 origins coincide at the shoulder height
    assert g.translation(frames["fr3_link2"]) == pytest.approx((0, 0, 0.333))
    assert g.translation(frames["fr3_link3"])[2] == pytest.approx(0.333 + 0.316)


def test_fk_rotation_about_joint1_is_a_yaw() -> None:
    k = FR3Kinematics()
    q = list(FR3_READY)
    q[0] = 0.5
    x0, y0, z0 = g.translation(k.fk(FR3_READY))
    x1, y1, z1 = g.translation(k.fk(q))
    assert math.hypot(x1, y1) == pytest.approx(math.hypot(x0, y0), abs=1e-9)
    assert math.atan2(y1, x1) == pytest.approx(0.5, abs=1e-9)
    assert z1 == pytest.approx(z0, abs=1e-12)


def test_fk_ik_round_trip() -> None:
    k = FR3Kinematics()
    rng = random.Random(42)
    for _ in range(6):
        q_true = [rng.uniform(lo + 0.3, hi - 0.3) for lo, hi in FR3_JOINT_LIMITS]
        target = k.fk(q_true)
        seed = [v + rng.uniform(-0.2, 0.2) for v in q_true]
        res = k.ik(target, seed)
        assert res.success, res
        assert all(lo <= v <= hi for v, (lo, hi) in zip(res.positions, FR3_JOINT_LIMITS, strict=True))
        reached = k.fk(res.positions)
        assert g.distance(g.translation(reached), g.translation(target)) < 1e-3
        assert math.sqrt(sum(c * c for c in g.rotation_error(reached, target))) < 5e-3


def test_ik_unreachable_reports_failure() -> None:
    k = FR3Kinematics()
    target = g.matrix_from_quat((1, 0, 0, 0), (2.0, 0.0, 0.5))  # 2 m away: out of reach
    res = k.ik(target, FR3_READY, restarts=1, max_iters=50)
    assert not res.success and res.position_error_m > 0.5


def test_quaternion_round_trip() -> None:
    q = g.quat_normalize((0.1, -0.7, 0.2, 0.6))
    q2 = g.quat_from_matrix(g.matrix_from_quat(q))
    assert g.angle_between_quats(q, q2) < 1e-6
    q3 = g.quat_normalize((0.0, 0.0, 0.0, 1.0))
    mid = g.slerp(q3, (0.0, 0.0, 1.0, 0.0), 0.5)  # halfway through a pi yaw = pi/2 yaw
    assert g.angle_between_quats(mid, q3) == pytest.approx(math.pi / 2, abs=1e-9)
