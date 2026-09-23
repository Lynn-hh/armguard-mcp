# Changelog

All notable changes to this project will be documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public version. Alpha: tested against a simulated FR3 only.

### Added

- MCP server on the official MCP Python SDK v2 (`mcp` 2.2), supporting protocol revisions 2025-11-25 and
  2026-07-28, over stdio and streamable HTTP (`armguard-mcp` CLI).
- 22 tools in six groups: introspect, perception, motion, gripper, control, and safety (always on).
- Strict YAML safety policy (`examples/policies/fr3.yaml`, `readonly.yaml`) covering joint limits with a
  margin, workspace box, keep-out zones, velocity and acceleration scaling caps, per-plan joint-travel and
  Cartesian path-length caps, force and torque thresholds, gripper limits, controller and camera-topic
  allowlists, per-tool and global rate limits, tool-group allowlist, and dry-run.
- Motion as plan, then preview, then execute, with TTL-limited, single-use plan handles bound to the
  start state. The whole path is validated (joint-space sampling plus FK) and independently re-validated
  before execution.
- Hard and soft violations: hard ones can never be executed, and soft ones need human approval in
  `outside_envelope` mode.
- Human approval through MCP elicitation, using `Resolve`-injected parameters hidden from the input
  schema. Clients without elicitation are denied by default, and `reset_estop` always requires approval.
- Force/torque monitor during execution. A violation stops the motion, latches the software e-stop and
  invalidates every plan.
- Software e-stop with a latch, `stop_motion`, and approval-gated `reset_estop` and `error_recovery`.
- Append-only JSONL audit log with redaction of image payloads, plus `get_audit_tail`.
- Deterministic fake FR3 backend: FR3 kinematics, timed execution, virtual table, Franka Hand, generated
  images, fake `ros2_control`.
- Documentation: README, threat model, architecture, demo transcript. `scripts/gen_tool_table.py` and
  `scripts/demo_fake.py`.
- CI (unit tests on Python 3.10 and 3.12, ROS 2 Jazzy job), an experimental Docker sketch, and
  `server.json` for the MCP Registry.
- ROS 2 backend (`--backend ros2`, `Ros2BackendConfig` under the policy's `ros2:` section or
  `--ros2-config`). It covers MoveIt 2 planning and FK services, `FollowJointTrajectory` execution with
  cancel, `controller_manager`, tf2, a `WrenchStamped` force estimate, the franka_gripper actions (or
  `GripperCommand`), and the franka_hardware collision-behaviour service and error-recovery action.
  Camera snapshots come from `Image` or `CompressedImage` topics. Integration tests in `tests_ros/` run
  on ROS 2 Jazzy, including against a real MoveIt 2 `move_group`.

### Known limitations

- ROS 2 backend not yet run on a physical robot.
- HTTP transport without authentication.
- The workspace check covers the TCP point only.

[0.1.0]: https://github.com/Lynn-hh/armguard-mcp/releases/tag/v0.1.0
