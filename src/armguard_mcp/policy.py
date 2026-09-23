"""Safety policy: a strictly validated YAML file that the server enforces on every tool call.

The policy is the single source of truth for what the LLM may do. It is loaded once at
start-up and is immutable for the lifetime of the server (there is deliberately no tool to
change it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from armguard_mcp.backends.ros2_config import Ros2BackendConfig

ToolGroup = Literal["introspect", "perception", "motion", "gripper", "control", "safety"]
ApprovalClass = Literal[
    "execute", "switch_controllers", "reset_estop", "error_recovery", "set_collision_thresholds"
]

# Which MCP tools belong to which group. ``safety`` is always registered.
TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "introspect": (
        "get_robot_state",
        "get_safety_envelope",
        "lookup_transform",
        "list_controllers",
        "list_ros_graph",
        "get_audit_tail",
    ),
    "perception": ("camera_snapshot",),
    "motion": ("plan_to_joints", "plan_to_pose", "plan_cartesian_path", "execute_plan", "get_motion_status"),
    "gripper": ("gripper_move", "gripper_grasp", "gripper_home"),
    "control": ("switch_controllers", "set_collision_thresholds"),
    "safety": ("stop_motion", "estop", "reset_estop", "error_recovery", "get_safety_status"),
}
ALL_TOOLS: frozenset[str] = frozenset(t for tools in TOOL_GROUPS.values() for t in tools)


class PolicyError(ValueError):
    """Raised when a policy file cannot be loaded or fails validation."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JointLimit(_Strict):
    min: float = Field(description="[rad]")
    max: float = Field(description="[rad]")
    max_velocity: float = Field(gt=0, description="[rad/s]")
    max_acceleration: float = Field(default=10.0, gt=0, description="[rad/s^2]")

    @model_validator(mode="after")
    def _ordered(self) -> JointLimit:
        if not self.min < self.max:
            raise ValueError(f"joint limit min ({self.min}) must be < max ({self.max})")
        return self


class RobotSection(_Strict):
    name: str
    planning_group: str
    base_frame: str
    ee_frame: str
    joint_names: list[str] = Field(min_length=1)
    joint_limits: dict[str, JointLimit]
    home_joint_positions: list[float]

    @model_validator(mode="after")
    def _consistent(self) -> RobotSection:
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("robot.joint_names contains duplicates")
        missing = [j for j in self.joint_names if j not in self.joint_limits]
        extra = [j for j in self.joint_limits if j not in self.joint_names]
        if missing or extra:
            raise ValueError(
                f"robot.joint_limits must cover exactly joint_names (missing={missing}, extra={extra})"
            )
        if len(self.home_joint_positions) != len(self.joint_names):
            raise ValueError("robot.home_joint_positions must have one entry per joint")
        for name, q in zip(self.joint_names, self.home_joint_positions, strict=True):
            lim = self.joint_limits[name]
            if not lim.min <= q <= lim.max:
                raise ValueError(
                    f"home position of {name} ({q}) is outside its limits [{lim.min}, {lim.max}]"
                )
        return self

    def limits_list(self) -> list[JointLimit]:
        return [self.joint_limits[j] for j in self.joint_names]


class ToolsSection(_Strict):
    enabled: list[ToolGroup] = Field(default_factory=lambda: ["introspect", "safety"])

    @field_validator("enabled")
    @classmethod
    def _safety_always(cls, v: list[str]) -> list[str]:
        # stop_motion / estop can never be disabled.
        out = list(dict.fromkeys(v))
        if "safety" not in out:
            out.append("safety")
        return out


class Box(_Strict):
    min: tuple[float, float, float] = Field(description="[m] in the robot base frame")
    max: tuple[float, float, float] = Field(description="[m] in the robot base frame")

    @model_validator(mode="after")
    def _ordered(self) -> Box:
        if not all(a < b for a, b in zip(self.min, self.max, strict=True)):
            raise ValueError(f"box min {self.min} must be < max {self.max} on every axis")
        return self

    def contains(self, p: tuple[float, float, float] | list[float], margin: float = 0.0) -> bool:
        return all(lo - margin <= c <= hi + margin for c, lo, hi in zip(p, self.min, self.max, strict=True))


class KeepOutZone(Box):
    name: str


class WorkspaceSection(_Strict):
    box: Box
    keep_out: list[KeepOutZone] = Field(default_factory=list)


class MotionSection(_Strict):
    max_velocity_scaling: float = Field(gt=0, le=1)
    max_acceleration_scaling: float = Field(gt=0, le=1)
    default_velocity_scaling: float = Field(default=0.1, gt=0, le=1)
    default_acceleration_scaling: float = Field(default=0.1, gt=0, le=1)
    max_joint_step_rad: float = Field(gt=0, description="Hard cap on any joint's travel in one plan")
    soft_joint_step_rad: float | None = Field(
        default=None,
        gt=0,
        description="Above this travel a plan counts as outside the envelope (needs approval)",
    )
    max_cartesian_step_m: float = Field(gt=0, description="Hard cap on TCP path length of a Cartesian plan")
    plan_ttl_s: float = Field(default=120.0, gt=0)
    joint_limit_margin_rad: float = Field(default=0.05, ge=0)
    start_tolerance_rad: float = Field(
        default=0.01, gt=0, description="Max drift from plan start before 'stale'"
    )
    check_resolution_rad: float = Field(
        default=0.02, gt=0, description="Joint-space sampling for envelope checks"
    )
    cartesian_eef_step_m: float = Field(default=0.005, gt=0)

    @model_validator(mode="after")
    def _defaults_within_caps(self) -> MotionSection:
        if self.default_velocity_scaling > self.max_velocity_scaling:
            raise ValueError("motion.default_velocity_scaling must be <= max_velocity_scaling")
        if self.default_acceleration_scaling > self.max_acceleration_scaling:
            raise ValueError("motion.default_acceleration_scaling must be <= max_acceleration_scaling")
        if self.soft_joint_step_rad is not None and self.soft_joint_step_rad > self.max_joint_step_rad:
            raise ValueError("motion.soft_joint_step_rad must be <= max_joint_step_rad")
        return self


class ForceSection(_Strict):
    max_contact_force_n: float = Field(gt=0)
    max_contact_torque_nm: float = Field(gt=0)
    monitor_rate_hz: float = Field(default=200.0, gt=0, le=5000)


class GripperSection(_Strict):
    min_width_m: float = Field(default=0.0, ge=0)
    max_width_m: float = Field(gt=0)
    max_grasp_force_n: float = Field(gt=0)
    max_speed_mps: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> GripperSection:
        if not self.min_width_m < self.max_width_m:
            raise ValueError("gripper.min_width_m must be < max_width_m")
        return self


class ControllersSection(_Strict):
    allowlist: list[str] = Field(default_factory=list)


class PerceptionSection(_Strict):
    camera_topics: list[str] = Field(default_factory=list)
    max_image_width: int = Field(default=640, ge=16, le=4096)


class RateLimitsSection(_Strict):
    global_per_minute: int = Field(default=120, ge=1)
    default_per_minute: int = Field(default=60, ge=1)
    per_tool: dict[str, int] = Field(default_factory=dict)

    @field_validator("per_tool")
    @classmethod
    def _known_tools(cls, v: dict[str, int]) -> dict[str, int]:
        unknown = sorted(set(v) - ALL_TOOLS)
        if unknown:
            raise ValueError(f"rate_limits.per_tool has unknown tools: {unknown}")
        bad = {k: n for k, n in v.items() if n < 1}
        if bad:
            raise ValueError(f"rate_limits.per_tool values must be >= 1: {bad}")
        return v


class ApprovalSection(_Strict):
    mode: Literal["always", "outside_envelope", "never"] = "outside_envelope"
    on_client_without_elicitation: Literal["deny", "allow"] = "deny"
    require_for: list[ApprovalClass] = Field(
        default_factory=lambda: [
            "execute",
            "switch_controllers",
            "reset_estop",
            "error_recovery",
            "set_collision_thresholds",
        ]
    )

    @field_validator("require_for")
    @classmethod
    def _reset_always(cls, v: list[str]) -> list[str]:
        # Resetting a software e-stop always needs a human, whatever the policy says.
        out = list(dict.fromkeys(v))
        if "reset_estop" not in out:
            out.append("reset_estop")
        return out


class Policy(_Strict):
    version: Literal[1] = 1
    robot: RobotSection
    tools: ToolsSection = Field(default_factory=ToolsSection)
    workspace: WorkspaceSection
    motion: MotionSection
    force: ForceSection
    gripper: GripperSection | None = None
    controllers: ControllersSection = Field(default_factory=ControllersSection)
    perception: PerceptionSection = Field(default_factory=PerceptionSection)
    rate_limits: RateLimitsSection = Field(default_factory=RateLimitsSection)
    approval: ApprovalSection = Field(default_factory=ApprovalSection)
    dry_run: bool = False
    ros2: Ros2BackendConfig | None = Field(
        default=None, description="Connection settings of the ros2 backend (ignored by other backends)"
    )

    @model_validator(mode="after")
    def _cross_checks(self) -> Policy:
        if "gripper" in self.tools.enabled and self.gripper is None:
            raise ValueError("tools.enabled contains 'gripper' but there is no gripper section")
        return self

    # --- convenience -------------------------------------------------------------------
    def enabled_tools(self) -> list[str]:
        return [t for group in self.tools.enabled for t in TOOL_GROUPS[group]]

    def with_dry_run(self, dry_run: bool = True) -> Policy:
        return self.model_copy(update={"dry_run": dry_run})

    @classmethod
    def from_dict(cls, data: Any, *, source: str = "<dict>") -> Policy:
        if not isinstance(data, dict):
            raise PolicyError(f"{source}: policy must be a YAML mapping, got {type(data).__name__}")
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            lines = [f"{source}: invalid policy ({e.error_count()} error(s))"]
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "<root>"
                lines.append(f"  - {loc}: {err['msg']}")
            raise PolicyError("\n".join(lines)) from e

    @classmethod
    def from_yaml(cls, text: str, *, source: str = "<string>") -> Policy:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise PolicyError(f"{source}: YAML parse error: {e}") from e
        return cls.from_dict(data, source=source)


def load_policy(path: str | Path) -> Policy:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise PolicyError(f"cannot read policy file {p}: {e}") from e
    return Policy.from_yaml(text, source=str(p))


def load_ros2_config(path: str | Path) -> Ros2BackendConfig:
    """Load ros2 backend settings from a standalone YAML file (a mapping, optionally under ``ros2:``)."""
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise PolicyError(f"cannot read ros2 config {p}: {e}") from e
    if isinstance(data, dict) and set(data) == {"ros2"}:
        data = data["ros2"]
    if not isinstance(data, dict):
        raise PolicyError(f"{p}: ros2 config must be a YAML mapping")
    try:
        return Ros2BackendConfig.model_validate(data)
    except ValidationError as e:
        lines = [f"{p}: invalid ros2 config ({e.error_count()} error(s))"]
        lines += [
            f"  - {'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()
        ]
        raise PolicyError("\n".join(lines)) from e
