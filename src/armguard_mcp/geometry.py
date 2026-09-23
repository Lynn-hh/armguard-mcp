"""Small pure-Python rigid-body helpers (no numpy dependency).

Matrices are 4x4 homogeneous transforms stored as nested lists (row-major).
Quaternions are (x, y, z, w) tuples, matching ROS ``geometry_msgs/Quaternion``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Mat4 = list[list[float]]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def identity() -> Mat4:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def matmul(a: Mat4, b: Mat4) -> Mat4:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def rot_x(angle: float) -> Mat4:
    c, s = math.cos(angle), math.sin(angle)
    return [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]


def rot_z(angle: float) -> Mat4:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def trans(x: float, y: float, z: float) -> Mat4:
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def invert(t: Mat4) -> Mat4:
    """Inverse of a rigid transform (rotation transpose, rotated negated translation)."""
    r_t = [[t[j][i] for j in range(3)] for i in range(3)]
    p = [t[0][3], t[1][3], t[2][3]]
    new_p = [-sum(r_t[i][k] * p[k] for k in range(3)) for i in range(3)]
    return [
        [r_t[0][0], r_t[0][1], r_t[0][2], new_p[0]],
        [r_t[1][0], r_t[1][1], r_t[1][2], new_p[1]],
        [r_t[2][0], r_t[2][1], r_t[2][2], new_p[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation(t: Mat4) -> Vec3:
    return (t[0][3], t[1][3], t[2][3])


def rotation(t: Mat4) -> list[list[float]]:
    return [row[:3] for row in t[:3]]


def quat_normalize(q: Sequence[float]) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_from_matrix(t: Sequence[Sequence[float]]) -> Quat:
    """Rotation part of ``t`` as a unit quaternion (x, y, z, w) with w >= 0."""
    m00, m01, m02 = t[0][0], t[0][1], t[0][2]
    m10, m11, m12 = t[1][0], t[1][1], t[1][2]
    m20, m21, m22 = t[2][0], t[2][1], t[2][2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = quat_normalize((x, y, z, w))
    if q[3] < 0:
        q = (-q[0], -q[1], -q[2], -q[3])
    return q


def matrix_from_quat(q: Sequence[float], p: Sequence[float] = (0.0, 0.0, 0.0)) -> Mat4:
    x, y, z, w = quat_normalize(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(p[0])],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(p[1])],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(p[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_error(current: Mat4, target: Mat4) -> Vec3:
    """Axis-angle vector (in the base frame) rotating ``current`` onto ``target``."""
    rc, rt = rotation(current), rotation(target)
    # R_err = R_t * R_c^T
    r = [[sum(rt[i][k] * rc[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    qx, qy, qz, qw = quat_from_matrix(r)
    sin_half = math.sqrt(qx * qx + qy * qy + qz * qz)
    if sin_half < 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(sin_half, qw)
    k = angle / sin_half
    return (qx * k, qy * k, qz * k)


def angle_between_quats(a: Sequence[float], b: Sequence[float]) -> float:
    qa, qb = quat_normalize(a), quat_normalize(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb, strict=True)))
    return 2.0 * math.acos(min(1.0, dot))


def slerp(a: Sequence[float], b: Sequence[float], t: float) -> Quat:
    qa, qb = list(quat_normalize(a)), list(quat_normalize(b))
    dot = sum(x * y for x, y in zip(qa, qb, strict=True))
    if dot < 0.0:
        qb = [-c for c in qb]
        dot = -dot
    if dot > 0.9995:
        return quat_normalize([x + t * (y - x) for x, y in zip(qa, qb, strict=True)])
    theta0 = math.acos(dot)
    theta = theta0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return quat_normalize([s0 * x + s1 * y for x, y in zip(qa, qb, strict=True)])


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``a @ x = b`` by Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(b)
    m = [[*row[:], b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-15:
            raise ValueError("singular matrix")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x
