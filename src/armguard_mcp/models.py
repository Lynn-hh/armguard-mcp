"""Pydantic data models shared by backends, the safety layer and MCP tools.

Units: metres, radians, seconds, newtons, newton-metres. Poses are expressed in the frame
named by ``frame_id``; quaternions are (x, y, z, w) like ROS.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Vector3(_Model):
    """A 3-vector. Its unit is given by the field that holds it ([m] for positions, [N] / [N*m] in wrenches)."""

    x: float = Field(description="x component")
    y: float = Field(description="y component")
    z: float = Field(description="z component")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @classmethod
    def of(cls, v: Any) -> Vector3:
        return cls(x=float(v[0]), y=float(v[1]), z=float(v[2]))


class Quaternion(_Model):
    """Unit quaternion (x, y, z, w), ROS convention; the identity is (0, 0, 0, 1)."""

    x: float = Field(default=0.0, description="x (vector part)")
    y: float = Field(default=0.0, description="y (vector part)")
    z: float = Field(default=0.0, description="z (vector part)")
    w: float = Field(default=1.0, description="w (scalar part)")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    @classmethod
    def of(cls, q: Any) -> Quaternion:
        return cls(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


class Pose(_Model):
    frame_id: str = Field(description="Frame the pose is expressed in")
    position: Vector3
    orientation: Quaternion = Field(default_factory=Quaternion, description="Unit quaternion (x, y, z, w)")


class JointState(_Model):
    names: list[str]
    positions: list[float] = Field(description="[rad]")
    velocities: list[float] = Field(default_factory=list, description="[rad/s]")
    efforts: list[float] = Field(default_factory=list, description="[N*m]")
    stamp: float = Field(default=0.0, description="Seconds since epoch (backend clock)")


class Wrench(_Model):
    frame_id: str
    force: Vector3 = Field(description="[N]")
    torque: Vector3 = Field(description="[N*m]")
    stamp: float = 0.0

    @property
    def force_norm(self) -> float:
        return self.force.norm()

    @property
    def torque_norm(self) -> float:
        return self.torque.norm()


PlanKind = Literal["joints", "pose", "cartesian"]


class Plan(_Model):
    """A time-parameterised joint trajectory produced by a backend planner."""

    plan_id: str
    kind: PlanKind
    joint_names: list[str]
    waypoints: list[list[float]] = Field(description="Joint positions per waypoint [rad]; waypoint 0 = start")
    time_from_start: list[float] = Field(description="Time of each waypoint [s]")
    duration_s: float
    created_at: float = Field(description="Epoch seconds")
    source_request: str = Field(default="", description="Short summary of the request that produced the plan")
    velocity_scaling: float = 1.0
    acceleration_scaling: float = 1.0

    @property
    def start(self) -> list[float]:
        return self.waypoints[0]

    @property
    def final(self) -> list[float]:
        return self.waypoints[-1]


Severity = Literal["hard", "soft"]


class Violation(_Model):
    code: str = Field(description="Machine-readable code, e.g. JOINT_LIMIT, WORKSPACE, KEEP_OUT")
    severity: Severity = Field(description="hard = never executable; soft = needs human approval")
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class EnvelopeVerdict(_Model):
    ok: bool = Field(description="False if any hard violation exists (plan can never be executed)")
    inside_envelope: bool = Field(description="True if there are no violations at all")
    violations: list[Violation] = Field(default_factory=list)

    @classmethod
    def from_violations(cls, violations: list[Violation]) -> EnvelopeVerdict:
        return cls(
            ok=not any(v.severity == "hard" for v in violations),
            inside_envelope=not violations,
            violations=violations,
        )

    @property
    def hard(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "hard"]

    @property
    def soft(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "soft"]


class PlanSummary(_Model):
    """What the LLM sees after planning. The full trajectory stays server-side."""

    plan_id: str
    kind: PlanKind
    status: Literal["executable", "needs_approval", "rejected"] = Field(
        description="rejected plans can never be executed; needs_approval requires a human to approve"
    )
    executable: bool
    duration_s: float
    num_waypoints: int
    start_joint_positions: list[float]
    final_joint_positions: list[float]
    final_ee_pose: Pose
    max_joint_velocity_ratio: float = Field(description="Peak joint speed as a fraction of the joint limit")
    max_joint_acceleration_ratio: float = Field(
        description="Peak joint acceleration (estimated from the waypoint timing) as a fraction of the joint limit"
    )
    max_joint_travel_rad: float
    tcp_path_length_m: float
    velocity_scaling: float
    acceleration_scaling: float
    violations: list[Violation] = Field(default_factory=list)
    requires_approval: bool
    expires_at: str | None = Field(default=None, description="ISO-8601 UTC expiry of the plan handle")
    dry_run: bool
    force_monitoring: Literal["available", "unavailable"] = Field(
        description="Whether the backend currently provides an external wrench estimate. Without one the "
        "server cannot enforce the contact force limit, and execute_plan refuses to move unless the "
        "policy sets force.require_wrench: false"
    )
    notes: list[str] = Field(default_factory=list)


class GripperState(_Model):
    width_m: float
    max_width_m: float
    is_grasped: bool
    stamp: float = 0.0


class ControllerInfo(_Model):
    name: str
    type: str
    state: str = Field(description="active | inactive | unconfigured | finalized")


class ControllerList(_Model):
    controllers: list[ControllerInfo]
    switchable: list[str] = Field(description="Controllers the policy allows to be (de)activated")


class GraphInfo(_Model):
    nodes: list[str]
    topics: list[str]
    services: list[str]
    actions: list[str]


class SafetyStatus(_Model):
    estopped: bool
    reason: str | None = None
    reason_source: Literal["agent", "server"] | None = Field(
        default=None, description="Who gave the reason: the AI agent (via estop) or the server itself"
    )
    estop_event: int | None = Field(default=None, description="Number of the latched e-stop event")
    estopped_at: str | None = None
    dry_run: bool
    force_violation_latched: bool
    last_violation: Violation | None = None
    executing_plan_id: str | None = None
    note: str = (
        "Software envelope only (defense in depth). The robot's own safety system and hardware "
        "e-stop remain the real safety layer."
    )


class RobotState(_Model):
    robot: str
    backend: str
    joint_state: JointState
    ee_pose: Pose
    wrench: Wrench | None = None
    gripper: GripperState | None = None
    safety: SafetyStatus


class ExecutionResult(_Model):
    success: bool
    message: str
    final_joint_state: JointState | None = None
    max_observed_force_n: float = 0.0
    aborted: bool = False


class ExecutionReport(_Model):
    plan_id: str
    status: Literal["completed", "aborted", "dry_run"]
    message: str
    executed: bool
    dry_run: bool
    approval: str = Field(description="How the execution was authorised")
    final_joint_positions: list[float] | None = None
    final_ee_pose: Pose | None = None
    max_observed_force_n: float | None = None
    duration_s: float


class MotionStatus(_Model):
    executing: bool
    plan_id: str | None = None
    progress: float = Field(default=0.0, description="0..1 fraction of the active plan")
    message: str = ""
    last_result: str | None = None


class ActionResult(_Model):
    """Generic result for state-changing tools."""

    ok: bool
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
