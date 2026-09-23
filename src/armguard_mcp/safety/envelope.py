"""Server-side safety envelope: pure, side-effect-free checks.

Hard violations (joint limits, workspace box, keep-out zones, step caps, speed caps) make a
plan permanently non-executable - not even a human approval can override them. Soft
conditions (close to a joint limit, unusually large motion) mark the plan as "outside the
envelope", which requires human approval when ``approval.mode == outside_envelope``.

The envelope checks the TCP point trajectory and joint-space quantities. It does NOT check
full link geometry for collisions; that is the planner's (MoveIt's) job.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from armguard_mcp.models import EnvelopeVerdict, Plan, Violation, Wrench
from armguard_mcp.policy import Policy

_EPS = 1e-9


def _v(code: str, severity: str, message: str, **detail: object) -> Violation:
    return Violation(code=code, severity=severity, message=message, detail=detail)  # type: ignore[arg-type]


def check_joint_positions(
    positions: Sequence[float], policy: Policy, *, where: str = "target", with_margin: bool = True
) -> list[Violation]:
    """Joint limits (hard) and proximity to limits within ``joint_limit_margin_rad`` (soft)."""
    robot = policy.robot
    if len(positions) != len(robot.joint_names):
        return [
            _v(
                "JOINT_COUNT",
                "hard",
                f"{where}: expected {len(robot.joint_names)} joint positions, got {len(positions)}",
            )
        ]
    out: list[Violation] = []
    margin = policy.motion.joint_limit_margin_rad
    for name, q in zip(robot.joint_names, positions, strict=True):
        lim = robot.joint_limits[name]
        if not math.isfinite(q):
            out.append(_v("NOT_FINITE", "hard", f"{where}: {name} is not a finite number", joint=name))
        elif q < lim.min - _EPS or q > lim.max + _EPS:
            out.append(
                _v(
                    "JOINT_LIMIT",
                    "hard",
                    f"{where}: {name}={q:.4f} rad is outside its limits [{lim.min}, {lim.max}]",
                    joint=name,
                    value=q,
                    min=lim.min,
                    max=lim.max,
                )
            )
        elif with_margin and margin > 0 and (q < lim.min + margin or q > lim.max - margin):
            out.append(
                _v(
                    "NEAR_JOINT_LIMIT",
                    "soft",
                    f"{where}: {name}={q:.4f} rad is within {margin} rad of a joint limit",
                    joint=name,
                    value=q,
                )
            )
    return out


def check_tcp_position(
    position: Sequence[float], policy: Policy, *, where: str = "target"
) -> list[Violation]:
    """Workspace box (hard) and keep-out zones (hard) for a TCP position in the base frame."""
    out: list[Violation] = []
    p = tuple(float(c) for c in position)
    ws = policy.workspace
    if not ws.box.contains(p):
        out.append(
            _v(
                "WORKSPACE",
                "hard",
                f"{where}: TCP ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m is outside the workspace box "
                f"{list(ws.box.min)}..{list(ws.box.max)}",
                position=list(p),
            )
        )
    for zone in ws.keep_out:
        if zone.contains(p):
            out.append(
                _v(
                    "KEEP_OUT",
                    "hard",
                    f"{where}: TCP ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m enters keep-out zone '{zone.name}'",
                    zone=zone.name,
                    position=list(p),
                )
            )
    return out


def densify(plan: Plan, resolution_rad: float) -> list[list[float]]:
    """Joint configurations sampled along the plan so consecutive samples differ by <= resolution."""
    samples: list[list[float]] = [list(plan.waypoints[0])]
    for a, b in zip(plan.waypoints, plan.waypoints[1:], strict=False):
        span = max((abs(y - x) for x, y in zip(a, b, strict=True)), default=0.0)
        n = max(1, math.ceil(span / resolution_rad))
        for i in range(1, n + 1):
            t = i / n
            samples.append([x + t * (y - x) for x, y in zip(a, b, strict=True)])
    return samples


def max_joint_velocity_ratio(plan: Plan, policy: Policy) -> float:
    limits = policy.robot.limits_list()
    ratio = 0.0
    for i in range(1, len(plan.waypoints)):
        dt = plan.time_from_start[i] - plan.time_from_start[i - 1]
        if dt <= 0:
            return math.inf
        for j, lim in enumerate(limits):
            ratio = max(ratio, abs(plan.waypoints[i][j] - plan.waypoints[i - 1][j]) / dt / lim.max_velocity)
    return ratio


def joint_travel(plan: Plan) -> list[float]:
    """Path length travelled by each joint over the whole plan [rad]."""
    n = len(plan.waypoints[0]) if plan.waypoints else 0
    travel = [0.0] * n
    for a, b in zip(plan.waypoints, plan.waypoints[1:], strict=False):
        for j in range(n):
            travel[j] += abs(b[j] - a[j])
    return travel


def path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(a, b) for a, b in itertools.pairwise(points))


def check_plan(
    plan: Plan,
    policy: Policy,
    samples: Sequence[Sequence[float]],
    tcp_positions: Sequence[Sequence[float]],
) -> EnvelopeVerdict:
    """Validate a plan against the envelope.

    ``samples`` are joint configurations along the plan (see :func:`densify`) and
    ``tcp_positions`` the corresponding TCP positions in the base frame (computed by the
    caller with the backend's forward kinematics), so this function stays pure.
    """
    robot, motion = policy.robot, policy.motion
    v: list[Violation] = []

    # Structural checks.
    if plan.joint_names != robot.joint_names:
        v.append(
            _v(
                "JOINT_NAMES",
                "hard",
                f"plan joints {plan.joint_names} do not match policy {robot.joint_names}",
            )
        )
        return EnvelopeVerdict.from_violations(v)
    if len(plan.waypoints) < 2 or len(plan.waypoints) != len(plan.time_from_start):
        v.append(_v("MALFORMED", "hard", "plan needs >= 2 waypoints and one timestamp per waypoint"))
        return EnvelopeVerdict.from_violations(v)
    if any(b <= a for a, b in zip(plan.time_from_start, plan.time_from_start[1:], strict=False)):
        v.append(_v("MALFORMED", "hard", "plan timestamps must be strictly increasing"))
        return EnvelopeVerdict.from_violations(v)
    if len(samples) != len(tcp_positions):
        raise ValueError("samples and tcp_positions must have equal length")

    # Joint limits along the whole path (hard) and near-limit at the goal (soft).
    seen_codes: set[tuple[str, str]] = set()
    for i, q in enumerate(samples):
        for viol in check_joint_positions(q, policy, where=f"sample {i}", with_margin=False):
            key = (viol.code, str(viol.detail.get("joint")))
            if key not in seen_codes:  # report each joint once, not once per sample
                seen_codes.add(key)
                v.append(viol)
    v.extend(x for x in check_joint_positions(plan.final, policy, where="goal") if x.severity == "soft")

    # TCP inside workspace box, outside keep-out zones, for every sample.
    seen_zone: set[str] = set()
    for i, p in enumerate(tcp_positions):
        for viol in check_tcp_position(p, policy, where=f"sample {i}"):
            key = viol.code + str(viol.detail.get("zone", ""))
            if key not in seen_zone:
                seen_zone.add(key)
                v.append(viol)

    # Joint travel caps.
    travel = joint_travel(plan)
    worst = max(travel) if travel else 0.0
    if worst > motion.max_joint_step_rad + _EPS:
        j = travel.index(worst)
        v.append(
            _v(
                "STEP_TOO_LARGE",
                "hard",
                f"{robot.joint_names[j]} would travel {worst:.3f} rad; the cap is {motion.max_joint_step_rad} rad "
                "per plan. Split the motion into smaller plans.",
                joint=robot.joint_names[j],
                travel=worst,
            )
        )
    elif motion.soft_joint_step_rad is not None and worst > motion.soft_joint_step_rad:
        v.append(
            _v(
                "LARGE_MOTION",
                "soft",
                f"largest joint travel {worst:.3f} rad exceeds the soft threshold {motion.soft_joint_step_rad} rad",
                travel=worst,
            )
        )

    # Cartesian path length cap.
    if plan.kind == "cartesian":
        length = path_length(tcp_positions)
        if length > motion.max_cartesian_step_m + 1e-6:
            v.append(
                _v(
                    "CARTESIAN_STEP_TOO_LARGE",
                    "hard",
                    f"TCP path length {length:.3f} m exceeds the {motion.max_cartesian_step_m} m cap per Cartesian plan",
                    length=length,
                )
            )

    # Speed cap: timing must respect max_velocity_scaling of the per-joint velocity limits.
    ratio = max_joint_velocity_ratio(plan, policy)
    if ratio > motion.max_velocity_scaling + 1e-6:
        v.append(
            _v(
                "VELOCITY",
                "hard",
                f"peak joint velocity is {ratio:.3f} of the limit; the policy cap is {motion.max_velocity_scaling}",
                ratio=ratio,
            )
        )
    if plan.velocity_scaling > motion.max_velocity_scaling + _EPS:
        v.append(_v("SCALING", "hard", f"velocity scaling {plan.velocity_scaling} exceeds the policy cap"))
    if plan.acceleration_scaling > motion.max_acceleration_scaling + _EPS:
        v.append(
            _v("SCALING", "hard", f"acceleration scaling {plan.acceleration_scaling} exceeds the policy cap")
        )

    return EnvelopeVerdict.from_violations(v)


def clamp_scaling(requested: float | None, default: float, maximum: float) -> tuple[float, str | None]:
    """Return (scaling, note). ``None`` -> default; above the cap -> clamped with a note."""
    if requested is None:
        return default, None
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError(f"scaling must be in (0, 1], got {requested}")
    if requested > maximum:
        return maximum, f"requested scaling {requested} clamped to the policy cap {maximum}"
    return requested, None


def check_wrench(wrench: Wrench, policy: Policy) -> list[Violation]:
    """External force/torque thresholds (hard)."""
    out: list[Violation] = []
    f, t = wrench.force_norm, wrench.torque_norm
    if f > policy.force.max_contact_force_n:
        out.append(
            _v(
                "FORCE_LIMIT",
                "hard",
                f"contact force {f:.1f} N exceeds the {policy.force.max_contact_force_n} N limit",
                force_n=f,
            )
        )
    if t > policy.force.max_contact_torque_nm:
        out.append(
            _v(
                "TORQUE_LIMIT",
                "hard",
                f"contact torque {t:.2f} N*m exceeds the {policy.force.max_contact_torque_nm} N*m limit",
                torque_nm=t,
            )
        )
    return out
