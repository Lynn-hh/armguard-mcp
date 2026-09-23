# Demo: a scripted session with the fake FR3

This is **real output**, not a mock-up. It was produced by

```bash
python scripts/demo_fake.py            # --mode legacy / --mode 2026-07-28 give the same flow
```

The script plays the MCP client (standing in for the LLM) and a human operator who approves every
prompt. It talks to the real server over an in-memory MCP session (`mcp.client.Client(server)`), with the
fake FR3 backend and [examples/policies/fr3.yaml](../examples/policies/fr3.yaml). Results are abridged
to the relevant keys, and floats are rounded to four decimals (`--full` prints everything). Plan ids and
timestamps change on every run. Step 3 also makes one unprinted `get_robot_state` call to compute its
target, which is why it appears in the audit trail.

What to look for:

- **Step 2:** a small motion inside the envelope runs without a prompt, because the policy is
  `approval.mode: outside_envelope`. Reusing the plan id fails, because plan handles are single-use.
- **Step 3:** a 0.9 rad turn of joint 1 exceeds `soft_joint_step_rad: 0.8`. The plan is `needs_approval`,
  and the human sees the prompt shown below before anything moves.
- **Step 4:** a target near the camera mount is `rejected`, because the TCP path enters the `camera_mount`
  keep-out zone at sample 96. Calling `execute_plan` on it anyway is refused, and no approval could change
  that.
- **Step 5:** `estop` latches. Planning is refused until a human approves `reset_estop`.
- **Step 6:** every call, including every denial, is in the audit log, with the operator's name recorded
  for the human approval.

(While developing the script, the first draft of step 3 turned joint 1 by +1.0 rad instead of -0.9 rad.
The server rejected that plan: the TCP would have swept out of the workspace box at x = 0.148 m, just
behind the 0.15 m boundary. The envelope checks the whole path, not only the goal.)

```text
# armguard-mcp demo: fake FR3, policy fr3.yaml, protocol mode 'auto'

## 1. Look before moving

>>> get_robot_state({})
{
  "robot": "fr3",
  "backend": "fake",
  "ee_pose": {"frame_id": "fr3_link0", "position": {"x": 0.3069, "y": -0.0, "z": 0.4869}, "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}},
  "wrench": {"frame_id": "fr3_link0", "force": {"x": 0.0, "y": 0.0, "z": 0.0}, "torque": {"x": 0.0, "y": 0.0, "z": 0.0}, "stamp": 1790141359.0855},
  "safety": {"estopped": false, "reason": null, "reason_source": null, "estop_event": null, "estopped_at": null, "dry_run": false, "force_violation_latched": false, "last_violation": null, "executing_plan_id": null, "note": "Software envelope only (defense in depth). The robot's own safety system and hardware e-stop remain the real safety layer."}
}

## 2. Small move inside the envelope: no approval needed (approval.mode=outside_envelope)

>>> plan_to_pose({"position": {"x": 0.4, "y": 0.1, "z": 0.4}})
{
  "plan_id": "16885d46bc86",
  "kind": "pose",
  "status": "executable",
  "executable": true,
  "duration_s": 1.566,
  "max_joint_travel_rad": 0.3188,
  "max_joint_velocity_ratio": 0.1,
  "violations": [],
  "requires_approval": false
}

>>> execute_plan({"plan_id": "16885d46bc86"})
{
  "plan_id": "16885d46bc86",
  "status": "completed",
  "message": "trajectory complete",
  "executed": true,
  "approval": "policy",
  "max_observed_force_n": 0.0
}
(progress notifications received: 10, last=1.00)

# Plan handles are single-use:

>>> execute_plan({"plan_id": "16885d46bc86"})
ERROR: Error executing tool execute_plan: plan 16885d46bc86 cannot be used: already used (plan handles are single-use). Plan again.

## 3. Large motion (joint 1 turns -0.9 rad > soft_joint_step_rad 0.8): a human must approve

>>> plan_to_joints({"joint_positions": [-0.851, -0.4666, 0.1682, -2.3493, 0.0793, 1.8874, 0.9601]})
{
  "plan_id": "809a290ba6eb",
  "kind": "joints",
  "status": "needs_approval",
  "executable": true,
  "duration_s": 3.6097,
  "max_joint_travel_rad": 0.9,
  "max_joint_velocity_ratio": 0.1,
  "violations": [{"code": "LARGE_MOTION", "severity": "soft", "message": "largest joint travel 0.900 rad exceeds the soft threshold 0.8 rad", "detail": {"travel": 0.9}}],
  "requires_approval": true
}

>>> execute_plan({"plan_id": "809a290ba6eb"})

--- approval prompt shown to the human (MCP elicitation) ---
APPROVE ROBOT MOTION on 'fr3'? Plan 809a290ba6eb (joints: joints -> [-0.851, -0.4666, 0.1682, -2.3493, 0.0793, 1.8874, 0.9601]).
Duration 3.61 s, 74 waypoints, peak joint speed 10% of limit (cap 30%), largest joint travel 0.900 rad, TCP path 0.371 m.
Final TCP in fr3_link0: x=0.327 y=-0.251 z=0.400 m.
Force monitoring: on - aborts above 25.0 N / 5.0 N*m.
Soft warnings: largest joint travel 0.900 rad exceeds the soft threshold 0.8 rad.
Dry run: NO - THE ROBOT WILL MOVE.
Approve only if the workspace is clear and the hardware e-stop is within reach.
--- human ticks 'approve', operator='lynn' ---
{
  "plan_id": "809a290ba6eb",
  "status": "completed",
  "message": "trajectory complete",
  "executed": true,
  "approval": "human",
  "max_observed_force_n": 0.0
}
(progress notifications received: 11, last=1.00)

## 4. Unsafe request: TCP into the 'camera_mount' keep-out zone

>>> plan_to_pose({"position": {"x": 0.62, "y": 0.42, "z": 0.3}})
{
  "plan_id": "2f3f7d1d6688",
  "kind": "pose",
  "status": "rejected",
  "executable": false,
  "duration_s": 5.4374,
  "max_joint_travel_rad": 1.568,
  "max_joint_velocity_ratio": 0.1,
  "violations": [{"code": "KEEP_OUT", "severity": "hard", "message": "sample 96: TCP (0.634, 0.356, 0.319) m enters keep-out zone 'camera_mount'", "detail": {"zone": "camera_mount", "position": [0.6336, 0.3558, 0.3187]}}, {"code": "LARGE_MOTION", "severity": "soft", "message": "largest joint travel 1.568 rad exceeds the soft threshold 0.8 rad", "detail": {"travel": 1.568}}],
  "requires_approval": false
}

# Even if the model tries anyway, a rejected plan can never be executed:

>>> execute_plan({"plan_id": "2f3f7d1d6688"})
ERROR: Error executing tool execute_plan: refused: plan 2f3f7d1d6688 violates hard safety limits and can never be executed: sample 96: TCP (0.634, 0.356, 0.319) m enters keep-out zone 'camera_mount'

## 5. Software e-stop latches; motion tools are refused until a human resets it

>>> estop({"reason": "demo: operator saw something odd"})
{
  "estopped": true,
  "reason": "demo: operator saw something odd",
  "reason_source": "agent",
  "estop_event": 1,
  "estopped_at": "2026-09-23T05:29:19.518Z",
  "dry_run": false,
  "force_violation_latched": false,
  "last_violation": {"code": "KEEP_OUT", "severity": "hard", "message": "sample 96: TCP (0.634, 0.356, 0.319) m enters keep-out zone 'camera_mount'", "detail": {"zone": "camera_mount", "position": [0.6336, 0.3558, 0.3187]}},
  "executing_plan_id": null,
  "note": "Software envelope only (defense in depth). The robot's own safety system and hardware e-stop remain the real safety layer."
}

>>> plan_to_pose({"position": {"x": 0.4, "y": 0.0, "z": 0.4}})
ERROR: Error executing tool plan_to_pose: refused: software e-stop is active (demo: operator saw something odd); call reset_estop (requires human approval)

## 6. Audit trail (get_audit_tail; last 9 events before this call, abridged)
{"seq": 4, "event": "tool_call", "tool": "execute_plan", "outcome": "ok", "plan_id": "16885d46bc86", "approval": {"decision": "not_required", "via": "policy"}}
{"seq": 5, "event": "tool_call", "tool": "execute_plan", "outcome": "denied", "plan_id": "16885d46bc86"}
{"seq": 6, "event": "tool_call", "tool": "get_robot_state", "outcome": "ok"}
{"seq": 7, "event": "tool_call", "tool": "plan_to_joints", "outcome": "ok", "plan_id": "809a290ba6eb"}
{"seq": 8, "event": "tool_call", "tool": "execute_plan", "outcome": "ok", "plan_id": "809a290ba6eb", "approval": {"decision": "approved", "via": "human", "operator": "lynn", "required": true}}
{"seq": 9, "event": "tool_call", "tool": "plan_to_pose", "outcome": "ok", "plan_id": "2f3f7d1d6688"}
{"seq": 10, "event": "tool_call", "tool": "execute_plan", "outcome": "denied", "plan_id": "2f3f7d1d6688"}
{"seq": 11, "event": "estop", "tool": "estop"}
{"seq": 12, "event": "tool_call", "tool": "estop", "outcome": "ok"}
```
