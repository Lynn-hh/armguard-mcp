# Architecture

armguard-mcp has three layers: the **MCP wiring** (tool definitions and approval resolvers), the
**policy engine** (`ArmGuard`, which knows nothing about how MCP transports work), and the **robot
backend** (which knows nothing about MCP, policies or approvals). The safety envelope is pure functions
over plain data, so it can be tested without a server, a robot or ROS.

## Module map

```
src/armguard_mcp/
├── cli.py              argparse entry point (`armguard-mcp`): load policy, create backend, pick transport.
│                        Logs to stderr only. Exit code 2 on a bad policy or an unavailable backend.
├── __main__.py         `python -m armguard_mcp`
├── server.py           MCP wiring + the ArmGuard policy engine:
│                          ArmGuard.call()           rate limit + audit wrapper around every tool body
│                          ArmGuard.approval_*()     pure approval decision (resolver side)
│                          ArmGuard.check_approval() authoritative interpretation (tool-body side)
│                          ArmGuard.finalize_plan()  validate + summarise + store a plan
│                          ArmGuard.run_execution()  execute with the force monitor, latch on violation
│                          build() / build_server()  register the enabled tool groups
├── policy.py           Strict, frozen pydantic schema for the YAML policy; TOOL_GROUPS
├── models.py           Shared data models (Pose, JointState, Wrench, Plan, PlanSummary, Violation, …)
├── plans.py            PlanStore: TTL, single use, invalidation, eviction; max_deviation()
├── safety/
│   ├── envelope.py     Pure checks: joint limits, workspace box, keep-out zones, travel/path caps,
│   │                    velocity ratio, scaling clamp, wrench thresholds; densify()
│   ├── ratelimit.py    Token buckets (per tool + global); never limits stop_motion/estop/get_safety_status
│   ├── state.py        SafetyState: e-stop latch, force-violation latch, dry-run, active execution
│   └── audit.py        Append-only JSONL + in-memory ring buffer; redaction of image/binary payloads
├── backends/
│   ├── base.py         RobotBackend ABC (all async) + BackendError hierarchy + CameraFrame
│   ├── __init__.py     create_backend("fake" | "ros2"); imports the ros2 backend lazily
│   ├── fake.py         Deterministic simulated FR3 (kinematics, planner, timed execution, virtual
│   │                    table for contact forces, Franka Hand, generated images, fake ros2_control)
│   ├── ros2.py         ROS 2 backend (the only module that imports rclpy, lazily)
│   └── ros2_config.py  Ros2BackendConfig: the policy's `ros2:` section (no ROS imports)
├── kinematics.py       FR3 forward kinematics (modified DH) and numerical IK, pure Python
├── geometry.py         4×4 transforms and quaternions, pure Python (no numpy)
└── imaging.py          Pure-Python PNG encoder; optional Pillow downscaling
```

The package imports and runs **without `rclpy`**. `tests/test_stdio.py` checks that importing the
package loads no `rclpy` module and prints nothing to stdout.

## Request lifecycle

Every tool follows the same path. The order matters: cheap and absolute refusals come first, and the human
is asked last, only about something that would actually run.

```
client ──tools/call──▶ SDK ──▶ [approval resolver(s)]  ──▶ tool body (runs once)
                               pure, may run >1×           ArmGuard.call(): rate limit ─▶ start backend
                               - pre-checks                  ─▶ e-stop / "already executing" checks
                               - Elicit(...) or              ─▶ argument validation / allowlists
                                 PolicyDecision              ─▶ plan lookup + re-validation + staleness
                                                             ─▶ check_approval()  (interpret outcome)
                                                             ─▶ consume plan id ─▶ backend action
                                                             ─▶ audit record (ok / denied / error / …)
```

1. **Registration.** `build()` registers only the tool groups in `policy.tools.enabled`, plus `safety`.
   A disabled tool does not exist on the wire.
2. **Resolvers** (only for approval-gated tools: `execute_plan`, `switch_controllers`,
   `set_collision_thresholds`, `error_recovery`, `reset_estop`). The tool's `approval` parameter is
   declared as `Annotated[ElicitationResult[ApprovalForm], Resolve(resolver)]`, which keeps it out of
   the input schema. The resolver runs pre-checks and returns one of two things:
   - a `PolicyDecision` (no prompt needed: not required, denied by a pre-check, or a client that cannot
     elicit);
   - `Elicit(message, ApprovalForm)`, which makes the SDK send an elicitation request to the client.

   Decline and cancel reach the tool body as `DeclinedElicitation` / `CancelledElicitation`, so they can
   be audited.
3. **`ArmGuard.call(tool, args)`** is an async context manager around every body. It consumes a
   rate-limit token (unless the tool is exempt), starts the backend lazily on first use, and writes
   exactly one `tool_call` audit record. It maps exceptions to results:
   - `Denied` → outcome `denied`, and the error is visible to the model;
   - `ToolError` → outcome `error`;
   - `BackendError` → a `ToolError` that says "robot backend error: …";
   - `PlanError` → `Denied`;
   - cancellation → outcome `cancelled`, then re-raised.
4. **Body.** The body repeats every check that matters (see the next section), calls `check_approval`,
   and then acts.
5. **Execution** (`run_execution`) runs `backend.execute(plan, on_progress, should_abort)` and a force
   monitor task in one anyio task group:
   - `on_progress` forwards to `ctx.report_progress` (best effort);
   - `should_abort` is polled by the backend every control tick, and becomes true when `stop_motion`,
     `estop` or the monitor call `request_abort`;
   - a force violation calls `backend.stop()`, latches the e-stop, invalidates all plans, writes a
     `force_violation` audit event, and returns an error to the model.

   A `BackendError` inside the task group is captured and re-raised outside it, so the model sees the
   real message instead of a hidden `ExceptionGroup`.

## Why resolvers are side-effect free

Under MCP 2026-07-28, a tool call that needs input from the user is a **multi-round-trip request**
(MRTR). The server answers the first `tools/call` with an `InputRequiredResult`, which carries the
elicitation request in `input_requests` plus a sealed `request_state`. The client collects the answer and
sends the call again with `input_responses` and the state, echoed verbatim. The SDK re-derives the injected parameters when that second call arrives,
so **a resolver can run more than once for one logical call.** We observed two runs per call. The tool
body runs once, after the last round.

That fixes the design:

- **Resolvers only read state.** They check the plan store with `peek` (never `consume`), check the rate
  limiter with `would_allow` (never `acquire`), read the joint state, and build the prompt. They write no
  audit records, move nothing and consume nothing.
- **All side effects happen in the body**: consuming the plan id, the rate-limit token, the audit
  record, the motion.
- **The body does not trust the resolver's view.** Time passes between the resolver's pre-check and the
  body, possibly while a human reads the prompt. So the body re-checks the e-stop, whether a plan is
  already executing, the plan's usability, the envelope (a full re-validation) and staleness. If the
  resolver concluded that no approval was needed, but the body now finds one is needed, the call is
  denied: "approval became required while the request was in flight; retry".

Tests pin this down. Under `mode="2026-07-28"` the robot moves exactly once and the human is prompted
exactly once per call, and the approval tests run in `auto`, `legacy` and `2026-07-28` modes.

Two SDK details shape `server.py`:

- `server.py` deliberately does **not** use `from __future__ import annotations`. The SDK resolves tool
  and resolver type hints at registration time. The `Resolve(...)` markers refer to closures inside
  `build()`, and with string annotations the lookup fails silently: the `approval` parameter would then
  show up in the input schema. A test checks that it stays hidden.
- `MCPServer(request_state_security=RequestStateSecurity.ephemeral(ttl=300))` seals `requestState`
  with AES-GCM under a key that exists only in this process. A server restart therefore invalidates
  approval round trips that are still in flight, and that is the intended behaviour.

## Plan handles

The model never sends a trajectory. A `plan_*` tool:

1. clamps velocity and acceleration scaling to the policy caps;
2. converts poses to the base frame, using the backend's tf when the model gives a `frame_id`;
3. asks the backend for a `Plan`: joint waypoints with timestamps, scaling, and the start state;
4. **validates** it: `densify` at `check_resolution_rad`, FK for every sample, then `check_plan` produces
   hard and soft `Violation`s;
5. stores it in `PlanStore` as a `StoredPlan(plan, summary, verdict, expires_at)`, including when it is
   rejected. That lets `execute_plan` later explain exactly why it refuses;
6. returns a `PlanSummary`: `plan_id`, `status` (`executable` / `needs_approval` / `rejected`), duration,
   waypoint count, start and final joints, final TCP pose, peak velocity ratio, largest travel, TCP path
   length, violations, `requires_approval`, `expires_at`, `dry_run`, notes.

Properties of a handle:

| Property | Implementation |
|---|---|
| Unguessable enough for this purpose | `uuid4().hex[:12]` (48 random bits) |
| Short-lived | `expires_at = now + plan_ttl_s` (default 120 s) |
| Single-use | `PlanStore.consume` moves the id to a "gone" map with the reason; a second use fails with "already used" |
| Bound to the start state | `execute_plan` refuses if `max_deviation(current joints, plan.start) > start_tolerance_rad` |
| Revocable | `invalidate_all` on e-stop and on force violation |
| Bounded memory | At most 256 live plans (the oldest is evicted); the "gone" map is trimmed above 4096 entries |

## Concurrency model

- **One event loop.** The MCP server runs on anyio (asyncio). Every tool is `async def`, so no tool runs
  on a worker thread. Several tool calls can be in flight at once on one session, because the SDK dispatches
  requests concurrently. This has been verified with the in-memory transport. That is what lets `stop_motion` or `estop` interrupt a running
  `execute_plan`, and `test_stop_motion_aborts_in_progress_execution` depends on it.
- **One motion at a time.** `SafetyState.active` holds the executing plan. `execute_plan`, controller
  switches, threshold changes and error recovery are refused while it is set.
- **Cooperative abort.** Stopping is two-pronged: `ActiveExecution.request_abort()` sets a flag that the
  backend polls every tick through `should_abort()`, and `backend.stop()` is called directly as well. The
  first abort reason wins.
- **Thread-safe shared state.** `SafetyState`, `PlanStore`, `RateLimiter` and `AuditLogger` guard their
  state with `threading.Lock`, so a backend may call into them from another thread (such as an rclpy
  callback) without corrupting them.
- **The ROS 2 bridge.** `rclpy` has its own executor, which blocks in `spin()`. The ROS 2 backend runs a
  `MultiThreadedExecutor` on a dedicated daemon thread, in a private `rclpy.Context`, and bridges every
  service and action future back to the asyncio loop with `loop.call_soon_threadsafe`
  (`Ros2Backend._await`). Subscriptions (joint states, wrench, camera) write the
  latest message into lock-protected caches, which the async methods read. The event loop never blocks on
  ROS, and ROS callbacks never touch MCP objects. `rclpy` is imported only inside
  `armguard_mcp.backends.ros2`, when `--backend ros2` is selected.
- **Real time stays in `ros2_control`.** The server sends whole trajectories (FollowJointTrajectory in
  the ROS 2 backend) and gripper actions. The 1 kHz impedance and force loops run in the Franka controllers
  and are never driven by the LLM or by Python.

## Extending

- **A new backend:** subclass `RobotBackend`, raise `BackendError` subclasses whose messages make sense to
  the model, and add a branch in `create_backend`. Run it through the same server tests by building it in
  `tests/conftest.py:make_app`.
- **A new tool:** add it to `TOOL_GROUPS` in `policy.py` (the rate-limit schema validates tool names
  against that list), register it in the matching `_register_*` in `server.py` with honest annotations,
  wrap the body in `guard.call(...)`, and call `guard.require_motion_allowed()` if the tool actuates
  anything. If it needs approval, add an approval class and a pure resolver, and re-check everything in
  the body.
