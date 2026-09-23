"""Configuration of the ROS 2 backend (the optional ``ros2:`` section of a policy file).

This module is deliberately free of ROS imports so the policy can be validated anywhere.
Defaults follow franka_ros2 v3.x on ROS 2 Jazzy (checked against the franka_ros2 v3.5.3
sources): topic, service and action names are those of a non-namespaced ``franka_bringup``
launch plus MoveIt's ``move_group``. **Verify them on your setup** (``ros2 topic list``,
``ros2 action list``) before trusting the defaults, especially if you launch with a namespace.

Nothing in here widens the safety envelope: this section only says *where* things are.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# libfranka's example collision thresholds (lower == upper) per joint [N*m]. Verify for your cell.
FRANKA_DEFAULT_JOINT_TORQUE_THRESHOLDS = (20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0)


class Ros2BackendConfig(BaseModel):
    """Where the ROS 2 backend finds the robot's topics, services and actions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- node / middleware -----------------------------------------------------------------
    node_name: str = Field(default="armguard_mcp", description="Name of the backend's rclpy node")
    namespace: str = Field(default="", description="Namespace of the backend's node (not of the robot)")
    domain_id: int | None = Field(
        default=None, ge=0, le=232, description="ROS_DOMAIN_ID override (default: environment)"
    )
    use_sim_time: bool = False
    executor_threads: int = Field(default=4, ge=2, le=32)
    startup_timeout_s: float = Field(
        default=10.0, gt=0, description="DDS discovery warm-up: how long start() waits for each endpoint"
    )
    service_timeout_s: float = Field(default=5.0, gt=0, description="Timeout of ordinary service calls")

    # --- state -----------------------------------------------------------------------------
    joint_states_topic: str = "/joint_states"
    joint_state_max_age_s: float = Field(
        default=0.5,
        gt=0,
        description="Joint states older than this are stale (the backend refuses to use them)",
    )
    tf_timeout_s: float = Field(default=1.0, gt=0)
    wrench_topic: str | None = Field(
        default="/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        description="geometry_msgs/WrenchStamped estimate of the external wrench (null = none). franka_ros2 "
        "also publishes .../external_wrench_in_base_frame; force and torque norms are the same in both.",
    )
    wrench_max_age_s: float = Field(default=0.5, gt=0)
    require_wrench: bool = Field(
        default=True,
        description="If true, a missing or stale wrench is an error (so execute_plan refuses to move without "
        "force monitoring). If false, it is reported as 'no estimate' and motion runs unmonitored.",
    )

    # --- kinematics / planning (MoveIt 2 move_group services) ------------------------------
    fk_source: Literal["moveit", "fr3_analytic"] = Field(
        default="moveit",
        description="moveit = /compute_fk on the robot's URDF (authoritative); fr3_analytic = built-in FR3 "
        "model (fast, no MoveIt needed, only valid for an unmodified FR3 + Franka Hand)",
    )
    compute_fk_service: str = "/compute_fk"
    plan_service: str = "/plan_kinematic_path"
    cartesian_path_service: str = "/compute_cartesian_path"
    pipeline_id: str = Field(default="", description="MoveIt planning pipeline ('' = move_group default)")
    planner_id: str = Field(default="", description="Planner id ('' = pipeline default)")
    planning_attempts: int = Field(default=1, ge=1, le=100)
    planning_time_s: float = Field(default=5.0, gt=0, le=120)
    goal_joint_tolerance_rad: float = Field(default=1e-3, gt=0)
    goal_position_tolerance_m: float = Field(default=1e-3, gt=0)
    goal_orientation_tolerance_rad: float = Field(default=1e-2, gt=0)
    cartesian_jump_threshold: float = Field(
        default=0.0, ge=0, description="Relative jump threshold (0 = off)"
    )
    cartesian_revolute_jump_threshold_rad: float = Field(
        default=0.3, ge=0, description="Absolute per-step joint jump threshold (0 = off)"
    )
    cartesian_avoid_collisions: bool = True
    cartesian_min_fraction: float = Field(
        default=1.0, gt=0, le=1, description="Refuse Cartesian paths that achieve less than this fraction"
    )

    # --- execution -------------------------------------------------------------------------
    trajectory_action: str = "/fr3_arm_controller/follow_joint_trajectory"
    goal_time_tolerance_s: float = Field(default=0.5, ge=0)
    execution_timeout_margin_s: float = Field(
        default=5.0, gt=0, description="Wait at most plan duration + this for the controller's result"
    )
    start_tolerance_rad: float = Field(
        default=0.02, gt=0, description="Backend-side check that the robot is at the plan's start state"
    )
    poll_period_s: float = Field(default=0.01, gt=0, le=0.1, description="Abort/progress polling period")

    # --- controllers -----------------------------------------------------------------------
    controller_manager: str = "/controller_manager"
    switch_strictness: Literal["strict", "best_effort"] = "strict"
    switch_timeout_s: float = Field(default=5.0, gt=0)

    # --- gripper ---------------------------------------------------------------------------
    gripper_interface: Literal["auto", "franka", "gripper_command", "none"] = Field(
        default="auto",
        description="auto = franka_msgs actions if franka_msgs is importable, else control_msgs/GripperCommand",
    )
    gripper_namespace: str = Field(
        default="/franka_gripper", description="franka_gripper node: <ns>/move, <ns>/grasp, <ns>/homing"
    )
    gripper_command_action: str = "/franka_gripper/gripper_action"
    gripper_command_position_is_half_width: bool = Field(
        default=True,
        description="franka_gripper's GripperCommand server moves to 2 * command.position; set false for "
        "grippers whose command.position is the full opening width",
    )
    gripper_joint_states_topic: str | None = "/franka_gripper/joint_states"
    gripper_width_scale: float = Field(
        default=1.0, gt=0, description="width = scale * sum(finger joint positions) (Franka Hand: 1.0)"
    )
    gripper_timeout_s: float = Field(default=30.0, gt=0)

    # --- perception ------------------------------------------------------------------------
    camera_timeout_s: float = Field(default=2.0, gt=0)

    # --- Franka-specific services / actions (null = not supported) -------------------------
    collision_behavior_service: str | None = "/service_server/set_force_torque_collision_behavior"
    collision_joint_torque_thresholds_nm: tuple[float, ...] = Field(
        default=FRANKA_DEFAULT_JOINT_TORQUE_THRESHOLDS,
        description="Per-joint torque thresholds sent with set_collision_thresholds (Franka needs all four "
        "threshold arrays in one call)",
    )
    error_recovery_action: str | None = "/action_server/error_recovery"
    error_recovery_timeout_s: float = Field(default=30.0, gt=0)

    @field_validator("collision_joint_torque_thresholds_nm")
    @classmethod
    def _seven(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if len(v) != 7 or any(x <= 0 for x in v):
            raise ValueError("collision_joint_torque_thresholds_nm needs 7 positive values")
        return v
