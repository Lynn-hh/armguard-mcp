# Bring-up on a physical Franka FR3

A staged checklist for the first session of armguard-mcp on a real FR3 with `franka_ros2` and MoveIt 2.
Each stage adds one capability; do not skip a stage. Budget 60 to 90 minutes including the checks.

**Before you start.** Get your lab's safety sign-off. One person holds the hardware e-stop for the whole
session and watches the robot, not the screen. Clear the workspace of anything you care about. Keep the
default speed caps; the policy below runs at 10% of the joint velocity limits.

## Stage 0. What the ROS graph looks like

With the robot powered, `franka_bringup` and `franka_fr3_moveit_config` launched (no namespace), in a
terminal that has sourced the workspace:

```bash
ros2 topic list | grep -E 'joint_states|external_wrench|image'
ros2 action list | grep -E 'follow_joint_trajectory|gripper|error_recovery'
ros2 service list | grep -E 'compute_fk|plan_kinematic_path|compute_cartesian_path|controller_manager|collision_behavior'
ros2 topic hz /franka_robot_state_broadcaster/external_wrench_in_stiffness_frame   # expect ~1000 Hz, ctrl-c
ros2 control list_controllers                                                     # fr3_arm_controller active
```

Compare every name with [examples/ros2/fr3_franka_ros2_jazzy.yaml](../examples/ros2/fr3_franka_ros2_jazzy.yaml).
A namespace prefixes all of them; put the real names in a copy of that file. If `franka_msgs` is not
built, the gripper falls back to `GripperCommand` and `set_collision_thresholds` / `error_recovery`
report "not supported", which is fine for this session.

Record: the robot's current joint positions (`ros2 topic echo /joint_states --once`) and the TCP position
from RViz or `ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp`. The policy's workspace box below is
written around that pose.

## Stage 1. Server up, nothing moves (`--dry-run`, read-only policy)

```bash
. .venv-ros/bin/activate                      # the venv that sees rclpy (README, "ROS 2 backend")
armguard-mcp --policy examples/policies/readonly.yaml --backend ros2 \
             --ros2-config my_fr3.yaml --audit-log bringup-$(date +%Y%m%d).jsonl --dry-run
```

It must start (exit status 2 means a missing endpoint: fix the name). From a second terminal, with
`examples/python_client.py` pointed at this command, or from Claude Code with the `.mcp.json` from the
README, call `get_robot_state`, `get_safety_envelope`, `list_controllers`, `lookup_transform` and
`camera_snapshot`. Check that the joint positions and TCP match stage 0 and that `force_monitoring` is
`available`. If it says `unavailable`, the wrench topic name or QoS is wrong; stop here until it is fixed,
because `execute_plan` will refuse to move without it.

## Stage 2. The session policy

Copy [examples/policies/fr3.yaml](../examples/policies/fr3.yaml) to `bringup.yaml` and tighten it to the
space in front of the robot. With the TCP at roughly (x, y, z) from stage 0:

```yaml
workspace:
  box: {min: [x-0.15, y-0.15, max(z-0.10, 0.05)], max: [x+0.15, y+0.15, z+0.20]}   # fill in numbers
  keep_out:
    - {name: table_edge, min: [x+0.10, y-0.30, 0.0], max: [x+0.40, y+0.30, z+0.30]}  # a box you can point at
motion:
  max_velocity_scaling: 0.1
  max_acceleration_scaling: 0.1
  default_velocity_scaling: 0.05
  default_acceleration_scaling: 0.05
  max_joint_step_rad: 0.4
  soft_joint_step_rad: 0.15       # anything above a small move asks the human
  max_cartesian_step_m: 0.10
force:
  max_contact_force_n: 15.0
  max_contact_torque_nm: 3.0
  require_wrench: true
rate_limits:
  per_tool: {execute_plan: 4}
approval:
  mode: outside_envelope
  on_client_without_elicitation: deny
```

Keep every other section as shipped. Start the server with `bringup.yaml`, still with `--dry-run`, and
run the five actions of stage 3 once in dry-run: every `execute_plan` should answer `dry_run` and nothing
should move. Then restart without `--dry-run`.

## Stage 3. Five actions, in this order

Use a client that shows elicitation prompts verbatim (Claude Code or Claude Desktop, or
`examples/python_client.py` with its `ask_human` callback). Say the robot is at the ready pose.

| # | Ask the agent | Expected | What to capture |
|---|---|---|---|
| 1 | "Where is the arm?" | `get_robot_state` only | nothing |
| 2 | "Move the gripper 5 cm up" | `plan_to_pose` → `executable`, `execute_plan` runs with no prompt at 5% speed; the arm moves 5 cm | photo or short video with the hardware e-stop in frame; the `execute_plan` audit line |
| 3 | "Move to a point inside the table_edge box" (give coordinates) | plan `rejected` with `KEEP_OUT`; `execute_plan` on it refused | the `plan_to_pose` and `execute_plan` audit lines |
| 4 | "Rotate joint 1 by 15 degrees" (0.26 rad > `soft_joint_step_rad`) | `needs_approval`; the prompt appears; approve once, decline once | the prompt text as shown, the approval audit line with your name |
| 5 | "Trigger the e-stop" then "Reset it" | planning refused while latched; `reset_estop` prompts; approve | the e-stop and reset audit lines |

Optional, only if you have a foam block and the sign-off for contact: "Move down 3 cm" with the block
under the gripper, to see the force monitor abort below 15 N. Note the force at which it aborted and how
long the stop took (`ros2 topic echo` timestamps). Skip it if in doubt.

## Stage 4. What to bring back

- `bringup-YYYYMMDD.jsonl` (the audit log). It is the evidence; images are already redacted from it.
- One photo of the arm mid-move with the hardware e-stop visible, or a 20-second video.
- The three names that differed from the example config, if any, and the measured wrench rate.
- Anything that surprised you: a refused call that should have run, a prompt you did not expect, a
  limit that felt wrong.

These go into [preventing-irreversible-actions.md](preventing-irreversible-actions.md) section 3.5 and
into the README's status paragraph, and the roadmap item "Bring-up on a physical FR3" gets checked off.
