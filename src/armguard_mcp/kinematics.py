"""Franka FR3 kinematics in pure Python (used by the fake backend).

Forward kinematics uses the modified (Craig) Denavit-Hartenberg parameters published by
Franka Robotics for the Panda/FR3 arm:

    joint  a [m]     d [m]   alpha [rad]
    1      0         0.333   0
    2      0         0       -pi/2
    3      0         0.316   pi/2
    4      0.0825    0       pi/2
    5      -0.0825   0.384   -pi/2
    6      0         0       pi/2
    7      0.088     0       pi/2
    flange 0         0.107   0

Each link transform is ``RotX(alpha) * TransX(a) * RotZ(q) * TransZ(d)``. The Franka Hand
TCP (``fr3_hand_tcp``) is the flange rotated by -pi/4 about z, then offset 0.1034 m along z.

Inverse kinematics is damped least squares on a finite-difference Jacobian, with joint-limit
clamping and deterministic restarts. It is intended for simulation and tests, not for
production planning (use MoveIt on the real robot).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from armguard_mcp import geometry as g

FR3_JOINT_NAMES: tuple[str, ...] = tuple(f"fr3_joint{i}" for i in range(1, 8))
FR3_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.7437, 2.7437),
    (-1.7837, 1.7837),
    (-2.9007, 2.9007),
    (-3.0421, -0.1518),
    (-2.8065, 2.8065),
    (0.5445, 4.5169),
    (-3.0159, 3.0159),
)
# Franka "ready" configuration.
FR3_READY: tuple[float, ...] = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4)

_DH_A = (0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088)
_DH_D = (0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0)
_DH_ALPHA = (0.0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2)
FLANGE_D = 0.107
HAND_TCP_Z = 0.1034
HAND_YAW = -math.pi / 4


@dataclass
class IKResult:
    success: bool
    positions: list[float]
    position_error_m: float
    orientation_error_rad: float
    iterations: int


@dataclass
class FR3Kinematics:
    """FR3 kinematic model. Frame names follow franka_description (prefix ``fr3``)."""

    prefix: str = "fr3"
    joint_limits: Sequence[tuple[float, float]] = field(default_factory=lambda: FR3_JOINT_LIMITS)

    @property
    def base_frame(self) -> str:
        return f"{self.prefix}_link0"

    @property
    def tcp_frame(self) -> str:
        return f"{self.prefix}_hand_tcp"

    def frame_names(self) -> list[str]:
        names = [f"{self.prefix}_link{i}" for i in range(0, 9)]
        return [*names, f"{self.prefix}_hand", f"{self.prefix}_hand_tcp"]

    def chain(self, q: Sequence[float]) -> dict[str, g.Mat4]:
        """Transforms of every frame of the arm expressed in the base frame."""
        if len(q) != 7:
            raise ValueError(f"expected 7 joint positions, got {len(q)}")
        frames: dict[str, g.Mat4] = {f"{self.prefix}_link0": g.identity()}
        t = g.identity()
        for i in range(7):
            link = g.matmul(g.matmul(g.rot_x(_DH_ALPHA[i]), g.trans(_DH_A[i], 0.0, 0.0)), g.rot_z(q[i]))
            link = g.matmul(link, g.trans(0.0, 0.0, _DH_D[i]))
            t = g.matmul(t, link)
            frames[f"{self.prefix}_link{i + 1}"] = t
        flange = g.matmul(t, g.trans(0.0, 0.0, FLANGE_D))
        frames[f"{self.prefix}_link8"] = flange
        hand = g.matmul(flange, g.rot_z(HAND_YAW))
        frames[f"{self.prefix}_hand"] = hand
        frames[f"{self.prefix}_hand_tcp"] = g.matmul(hand, g.trans(0.0, 0.0, HAND_TCP_Z))
        return frames

    def fk(self, q: Sequence[float]) -> g.Mat4:
        """Pose of the hand TCP in the base frame."""
        return self.chain(q)[self.tcp_frame]

    def jacobian(self, q: Sequence[float], eps: float = 1e-6) -> list[list[float]]:
        """6x7 geometric Jacobian by central finite differences (rows: vx vy vz wx wy wz)."""
        cols: list[list[float]] = []
        for j in range(7):
            qp, qm = list(q), list(q)
            qp[j] += eps
            qm[j] -= eps
            tp, tm = self.fk(qp), self.fk(qm)
            dp = [(tp[i][3] - tm[i][3]) / (2 * eps) for i in range(3)]
            dr = g.rotation_error(tm, tp)
            cols.append(dp + [c / (2 * eps) for c in dr])
        return [[cols[j][i] for j in range(7)] for i in range(6)]

    def clamp(self, q: Sequence[float]) -> list[float]:
        return [min(max(v, lo), hi) for v, (lo, hi) in zip(q, self.joint_limits, strict=True)]

    def ik(
        self,
        target: g.Mat4,
        seed: Sequence[float],
        *,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        max_iters: int = 150,
        restarts: int = 8,
        damping: float = 0.05,
    ) -> IKResult:
        """Damped-least-squares IK for the TCP pose ``target`` (base frame).

        Tries ``seed`` first, then deterministic pseudo-random seeds inside the joint limits.
        Returns the best solution found; ``success`` tells whether tolerances were met.
        """
        rng = random.Random(0)
        seeds: list[list[float]] = [self.clamp(seed)]
        for _ in range(restarts):
            seeds.append([rng.uniform(lo + 0.1, hi - 0.1) for lo, hi in self.joint_limits])
        best: IKResult | None = None
        total_iters = 0
        for s in seeds:
            q = list(s)
            for _ in range(max_iters):
                total_iters += 1
                cur = self.fk(q)
                dp = [target[i][3] - cur[i][3] for i in range(3)]
                dr = list(g.rotation_error(cur, target))
                pe = math.sqrt(sum(v * v for v in dp))
                re = math.sqrt(sum(v * v for v in dr))
                if pe < pos_tol and re < rot_tol:
                    return IKResult(True, q, pe, re, total_iters)
                err = dp + dr
                jac = self.jacobian(q)
                # dq = J^T (J J^T + lambda^2 I)^-1 e
                jjt = [
                    [
                        sum(jac[r][k] * jac[c][k] for k in range(7)) + (damping**2 if r == c else 0.0)
                        for c in range(6)
                    ]
                    for r in range(6)
                ]
                try:
                    y = g.solve_linear(jjt, err)
                except ValueError:
                    break
                dq = [sum(jac[r][k] * y[r] for r in range(6)) for k in range(7)]
                step = max(abs(v) for v in dq)
                if step > 0.2:  # limit per-iteration joint change for stability
                    dq = [v * 0.2 / step for v in dq]
                q = self.clamp([a + b for a, b in zip(q, dq, strict=True)])
            cur = self.fk(q)
            pe = g.distance(g.translation(cur), g.translation(target))
            re = math.sqrt(sum(v * v for v in g.rotation_error(cur, target)))
            cand = IKResult(pe < pos_tol and re < rot_tol, q, pe, re, total_iters)
            if cand.success:
                return cand
            if best is None or (pe + 0.1 * re) < (best.position_error_m + 0.1 * best.orientation_error_rad):
                best = cand
        assert best is not None
        return best
