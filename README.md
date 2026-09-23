# armguard-mcp

A safety-first [Model Context Protocol](https://modelcontextprotocol.io) server for ROS 2 manipulators.
An LLM agent can inspect the robot, plan motions, preview them, and execute them. Every action goes through
a **safety envelope that the server enforces outside the LLM**.

> **Status: alpha.** The core (policy, safety envelope, approval flow, audit log, MCP tools) runs against
> a simulated Franka FR3 backend. The ROS 2 / `rclpy` backend is not implemented yet.
>
> **This is defense in depth. It is NOT certified functional safety.** The robot's own safety system
> (for example the Franka safety configuration), a risk assessment under ISO 10218, and a reachable hardware
> e-stop are the real safety layer. armguard-mcp limits what an LLM can ask the robot to do. It cannot make
> an unsafe cell safe.

## Design principles

- **The LLM is never in the control loop.** MCP works at the level of skills and tasks: plan, preview,
  execute, grasp. Real-time control (1 kHz impedance and force control) stays in `ros2_control` and the
  Franka controllers.
- **Plan, then preview, then execute.** `plan_*` tools return a summary and a short-lived, single-use
  `plan_id`. Only `execute_plan(plan_id)` moves the robot. Before moving, the server re-validates the plan,
  checks that the robot has not moved since planning (a "stale" plan), and checks the e-stop and rate limits.
- **The server enforces the limits, not the prompt.** The YAML policy sets joint limits with a margin, a
  workspace box, keep-out zones, caps on velocity and acceleration scaling, caps on joint travel and on
  Cartesian path length per plan, force and torque thresholds, gripper limits, a controller allowlist, a
  camera-topic allowlist, per-tool and global rate limits, tool-group allowlists, and dry-run mode.
- **Hard limits cannot be overridden.** A plan that breaks a hard limit is never executable, even if a
  human approves it. Soft conditions, such as a joint near its limit or a large motion, need human approval.
- **Humans approve through MCP elicitation.** Approval uses a `Resolve`-injected parameter, so it works on
  both the 2025-11-25 and 2026-07-28 protocol revisions. Clients that cannot ask a human are denied by default.
- **Everything is audited.** An append-only JSONL log records every call, denial, approval, e-stop and force
  violation. Image payloads are redacted.

## Quick start (simulated FR3, no ROS needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
armguard-mcp --policy examples/policies/fr3.yaml --audit-log audit.jsonl          # stdio
armguard-mcp --policy examples/policies/fr3.yaml --transport http --port 8765     # streamable HTTP
pytest -q
```

The HTTP transport has no authentication in this build. Bind it to `127.0.0.1`, which is the default.

## Tools

| Group | Tools |
|---|---|
| introspect (read-only) | `get_robot_state`, `get_safety_envelope`, `lookup_transform`, `list_controllers`, `list_ros_graph`, `get_audit_tail` |
| perception (read-only) | `camera_snapshot` (topics must be allowlisted) |
| motion | `plan_to_joints`, `plan_to_pose`, `plan_cartesian_path`, `execute_plan`, `get_motion_status` |
| gripper | `gripper_move`, `gripper_grasp`, `gripper_home` |
| control | `switch_controllers`, `set_collision_thresholds` |
| safety (always on) | `stop_motion`, `estop`, `reset_estop`, `error_recovery`, `get_safety_status` |

`stop_motion`, `estop` and `get_safety_status` are never rate limited and never need approval.
`reset_estop` always needs human approval.

## Known limitations

- The envelope checks the TCP point along a densely sampled joint path. It does not check full link
  geometry against the environment; collision checking is MoveIt's job.
- Force monitoring uses the backend's estimated external wrench, polled at `force.monitor_rate_hz`. It is a
  software backstop, not a replacement for the robot's collision reflexes.
- The fake backend is kinematic only. It has no dynamics and no self-collision model.

## License

Apache-2.0
