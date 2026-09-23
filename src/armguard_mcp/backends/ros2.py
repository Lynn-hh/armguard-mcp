"""ROS 2 backend: MoveIt 2 services for planning, ros2_control for execution, franka_ros2 extras.

Importing this module never imports ``rclpy``. ROS packages are imported when a
:class:`Ros2Backend` is constructed, and a missing ``rclpy`` raises a ``RuntimeError`` that
tells the user to source a ROS 2 installation.

Threading model
    One rclpy node (with a ``ReentrantCallbackGroup``) spun by a ``MultiThreadedExecutor`` on
    a daemon thread, inside a private ``rclpy.Context`` (the backend never touches the global
    context and installs no signal handlers). ROS callbacks only update caches or hand results
    to the asyncio loop with ``loop.call_soon_threadsafe``; every public method is a coroutine
    on the MCP server's event loop. rclpy futures are awaited through :meth:`Ros2Backend._await`.

What the backend talks to (all names configurable, see ``Ros2BackendConfig``)
    - ``/joint_states`` (sensor_msgs/JointState) and tf2 for state,
    - MoveIt ``move_group``: ``/compute_fk``, ``/plan_kinematic_path``, ``/compute_cartesian_path``,
    - the arm's ``FollowJointTrajectory`` action (joint_trajectory_controller) for execution,
    - ``controller_manager`` list/switch services,
    - a ``WrenchStamped`` external-wrench estimate for the server's force monitor,
    - franka_gripper actions (or control_msgs/GripperCommand), franka_hardware's
      collision-behaviour service and error-recovery action, camera topics.

The server validates everything against the policy *before* calling the backend. The backend
executes exactly the joint trajectory that was validated (the ``Plan`` waypoints and times);
MoveIt's velocities/accelerations are attached only if they belong to the same points.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import importlib
import itertools
import logging
import math
import struct
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import anyio

from armguard_mcp import geometry as g
from armguard_mcp.backends.base import (
    AbortCheck,
    BackendFailed,
    BackendTimeout,
    CameraFrame,
    NotSupported,
    ProgressCallback,
    RobotBackend,
)
from armguard_mcp.backends.ros2_config import Ros2BackendConfig
from armguard_mcp.imaging import encode_png_rgb, pillow_available, png_size
from armguard_mcp.models import (
    ControllerInfo,
    ExecutionResult,
    GraphInfo,
    GripperState,
    JointState,
    Plan,
    PlanKind,
    Pose,
    Quaternion,
    Vector3,
    Wrench,
)
from armguard_mcp.plans import new_plan_id

__all__ = ["Ros2Backend", "Ros2BackendConfig"]

logger = logging.getLogger("armguard_mcp.ros2")

_PARAM_SERVICE_SUFFIXES = (
    "/describe_parameters",
    "/get_parameter_types",
    "/get_parameters",
    "/list_parameters",
    "/set_parameters",
    "/set_parameters_atomically",
    "/get_type_description",
)
_MAX_STORED_TRAJECTORIES = 64

# action_msgs/GoalStatus values (stable across distros)
_STATUS_SUCCEEDED, _STATUS_CANCELED, _STATUS_ABORTED = 4, 5, 6
_STATUS_NAMES = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",
    5: "CANCELED",
    6: "ABORTED",
}


# ======================================================================================
# ROS-free helpers (unit-tested without ROS)
# ======================================================================================
def duration_to_s(d: Any) -> float:
    """builtin_interfaces/Duration (or Time) -> seconds."""
    return float(d.sec) + float(d.nanosec) * 1e-9


def s_to_sec_nanosec(t: float) -> tuple[int, int]:
    sec = math.floor(t)
    nsec = round((t - sec) * 1e9)
    if nsec >= 1_000_000_000:
        sec, nsec = sec + 1, nsec - 1_000_000_000
    return int(sec), int(nsec)


@dataclass
class TrajectoryArrays:
    waypoints: list[list[float]]
    times: list[float]
    velocities: list[list[float]] | None
    accelerations: list[list[float]] | None


def trajectory_arrays(
    msg_joint_names: Sequence[str], points: Sequence[Any], joint_names: Sequence[str]
) -> TrajectoryArrays:
    """Re-order a trajectory_msgs/JointTrajectory to ``joint_names`` and validate its timing.

    Raises BackendFailed if joints are missing, if the planner moved joints the policy does not
    cover (they would escape the safety envelope), or if the trajectory is not time-parameterised.
    """
    names = list(msg_joint_names)
    missing = [j for j in joint_names if j not in names]
    if missing:
        raise BackendFailed(f"planner trajectory lacks joints {missing} (it has {names})")
    extra = [j for j in names if j not in joint_names]
    if extra:
        raise BackendFailed(
            f"planner trajectory also moves joints {extra} that the policy does not cover; set "
            "robot.planning_group to an arm-only MoveIt group"
        )
    if not points:
        raise BackendFailed("planner returned an empty trajectory")
    idx = [names.index(j) for j in joint_names]
    n = len(names)

    def pick(values: Sequence[float]) -> list[float]:
        return [float(values[i]) for i in idx]

    waypoints: list[list[float]] = []
    times: list[float] = []
    vel_rows: list[list[float]] = []
    acc_rows: list[list[float]] = []
    for k, p in enumerate(points):
        if len(p.positions) != n:
            raise BackendFailed(f"trajectory point {k} has {len(p.positions)} positions for {n} joints")
        q = pick(p.positions)
        if not all(math.isfinite(v) for v in q):
            raise BackendFailed(f"trajectory point {k} contains non-finite positions")
        waypoints.append(q)
        times.append(duration_to_s(p.time_from_start))
        if len(p.velocities) == n:
            vel_rows.append(pick(p.velocities))
        if len(p.accelerations) == n:
            acc_rows.append(pick(p.accelerations))
    # derivatives are only usable if every point has them
    vels = vel_rows if len(vel_rows) == len(waypoints) else None
    accs = acc_rows if len(acc_rows) == len(waypoints) else None
    if len(waypoints) == 1:  # already at the goal: make a valid, short hold trajectory
        waypoints.append(list(waypoints[0]))
        times = [0.0, max(times[0], 0.1)]
        vels = [[0.0] * len(joint_names)] * 2 if vels is not None else None
        accs = [[0.0] * len(joint_names)] * 2 if accs is not None else None
    if any(b < a for a, b in itertools.pairwise(times)):
        raise BackendFailed("planner trajectory has decreasing time_from_start")
    if times[-1] <= 0.0:
        raise BackendFailed("planner trajectory is not time-parameterised (duration 0)")
    return TrajectoryArrays(waypoints, times, vels, accs)


def compressed_image_info(data: bytes) -> tuple[str, int, int]:
    """(mime, width, height) of a PNG or baseline/progressive JPEG payload."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = png_size(data)
        return "image/png", w, h
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
                i += 1 if marker == 0xFF else 2
                continue
            (seg_len,) = struct.unpack(">H", data[i + 2 : i + 4])
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return "image/jpeg", w, h
            i += 2 + seg_len
        raise NotSupported("JPEG image without a SOF header")
    raise NotSupported("compressed image is neither PNG nor JPEG")


_CHANNELS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8UC1": 1, "8UC3": 3}


def raw_image_to_png(
    encoding: str, width: int, height: int, step: int, data: bytes, max_width: int
) -> tuple[bytes, int, int]:
    """Convert an 8-bit sensor_msgs/Image to PNG, shrinking it to at most ``max_width`` pixels wide.

    Uses Pillow (area resampling) when installed, otherwise integer decimation in pure Python.
    """
    ch = _CHANNELS.get(encoding)
    if ch is None:
        raise NotSupported(
            f"image encoding {encoding!r} is not supported (8-bit rgb/bgr/rgba/bgra/mono only)"
        )
    if width <= 0 or height <= 0 or step < width * ch or len(data) < step * height:
        raise BackendFailed("malformed image message (size/step do not match the data)")
    rows = [bytes(data[r * step : r * step + width * ch]) for r in range(height)]

    def to_rgb(row: bytes) -> bytes:
        if ch == 1:
            out = bytearray(len(row) * 3)
            out[0::3] = out[1::3] = out[2::3] = row
            return bytes(out)
        out = bytearray(width * 3)
        if encoding.startswith("bgr"):
            out[0::3], out[1::3], out[2::3] = row[2::ch], row[1::ch], row[0::ch]
        else:
            out[0::3], out[1::3], out[2::3] = row[0::ch], row[1::ch], row[2::ch]
        return bytes(out)

    rgb = [to_rgb(r) for r in rows]
    if pillow_available():
        import io

        from PIL import Image

        im = Image.frombytes("RGB", (width, height), b"".join(rgb))
        if width > max_width:
            im = im.resize((max_width, max(1, round(height * max_width / width))), Image.Resampling.BOX)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue(), im.width, im.height
    k = max(1, math.ceil(width / max_width))
    if k > 1:
        new_w = len(range(0, width, k))
        small = []
        for r in rgb[::k]:
            out = bytearray(new_w * 3)
            out[0::3], out[1::3], out[2::3] = r[0 :: 3 * k], r[1 :: 3 * k], r[2 :: 3 * k]
            small.append(bytes(out))
        rgb, width, height = small, new_w, len(small)
    return encode_png_rgb(width, height, rgb), width, height


def filter_graph(
    nodes: Sequence[str],
    topics: Sequence[str],
    services: Sequence[str],
    actions: Sequence[str],
    own_node: str,
) -> GraphInfo:
    """Drop hidden action plumbing, parameter services and the backend's own node from a graph listing."""
    return GraphInfo(
        nodes=sorted(n for n in set(nodes) if n != own_node),
        topics=sorted(t for t in set(topics) if "/_action/" not in t),
        services=sorted(
            s for s in set(services) if "/_action/" not in s and not s.endswith(_PARAM_SERVICE_SUFFIXES)
        ),
        actions=sorted(set(actions)),
    )


def _ros_modules() -> SimpleNamespace:
    """Import rclpy and the message packages. Optional packages come back as None."""
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
    except ImportError as e:
        raise RuntimeError(
            "the ros2 backend is unavailable: rclpy is not importable. Source a ROS 2 installation (e.g. `source "
            f"/opt/ros/jazzy/setup.bash`) and run armguard-mcp with that Python ({e})"
        ) from e

    def opt(name: str) -> Any:
        try:
            return importlib.import_module(name)
        except ImportError:
            return None

    ns = SimpleNamespace(
        rclpy=rclpy,
        ActionClient=ActionClient,
        ReentrantCallbackGroup=ReentrantCallbackGroup,
        MultiThreadedExecutor=MultiThreadedExecutor,
        Node=Node,
        Parameter=Parameter,
        qos_sensor=qos_profile_sensor_data,
        Time=Time,
        sensor_msgs=opt("sensor_msgs.msg"),
        geometry_msgs=opt("geometry_msgs.msg"),
        trajectory_msgs=opt("trajectory_msgs.msg"),
        builtin_interfaces=opt("builtin_interfaces.msg"),
        tf2_ros=opt("tf2_ros"),
        moveit_msg=opt("moveit_msgs.msg"),
        moveit_srv=opt("moveit_msgs.srv"),
        shape_msgs=opt("shape_msgs.msg"),
        control_action=opt("control_msgs.action"),
        cm_srv=opt("controller_manager_msgs.srv"),
        franka_action=opt("franka_msgs.action"),
        franka_msg=opt("franka_msgs.msg"),
        franka_srv=opt("franka_msgs.srv"),
    )
    for required in ("sensor_msgs", "geometry_msgs", "trajectory_msgs", "builtin_interfaces"):
        if getattr(ns, required) is None:
            raise RuntimeError(f"the ros2 backend needs the {required} Python package (part of ros-base)")
    try:
        from rclpy.signals import SignalHandlerOptions

        ns.SignalHandlerOptions = SignalHandlerOptions
    except ImportError:  # pragma: no cover - very old rclpy
        ns.SignalHandlerOptions = None
    return ns


def _moveit_code_name(moveit_msg: Any, val: int) -> str:
    cls = moveit_msg.MoveItErrorCodes
    for name in dir(type(cls)):
        if name.isupper() and getattr(cls, name, None) == val:
            return name
    return str(val)


@dataclass
class _Timed:
    value: Any
    received: float  # time.monotonic() at reception


# ======================================================================================
# Backend
# ======================================================================================
class Ros2Backend(RobotBackend):
    """Talks to a real (or simulated) ROS 2 manipulator. See the module docstring."""

    name = "ros2"

    def __init__(
        self,
        config: Ros2BackendConfig,
        *,
        joint_names: Sequence[str],
        base_frame: str,
        ee_frame: str,
        planning_group: str,
        gripper_max_width_m: float = 0.08,
    ) -> None:
        self.cfg = config
        self.R = _ros_modules()
        self.joint_names = list(joint_names)
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.planning_group = planning_group
        self.gripper_max_width = gripper_max_width_m
        self._kin: Any = None
        if config.fk_source == "fr3_analytic":
            from armguard_mcp.kinematics import FR3Kinematics

            prefix = base_frame.rsplit("_link0", 1)[0]
            kin = FR3Kinematics(prefix=prefix)
            if (
                base_frame != kin.base_frame
                or ee_frame not in kin.frame_names()
                or len(self.joint_names) != 7
            ):
                raise ValueError(
                    "fk_source=fr3_analytic needs an FR3 (7 joints, base <prefix>_link0, ee one of "
                    f"{kin.frame_names()}); got base={base_frame!r} ee={ee_frame!r}"
                )
            self._kin = kin
        interface = config.gripper_interface
        if interface == "auto":
            interface = "franka" if self.R.franka_action is not None else "gripper_command"
        if interface == "franka" and self.R.franka_action is None:
            raise RuntimeError(
                "gripper_interface=franka but franka_msgs is not importable (build franka_ros2)"
            )
        if interface == "gripper_command" and self.R.control_action is None:
            interface = "none"
        self.gripper_interface = interface

        self._lock = threading.Lock()
        self._joints: dict[str, tuple[float, float, float, float]] = {}  # name -> (pos, vel, eff, recv)
        self._joint_stamp = 0.0
        self._wrench: _Timed | None = None
        self._gripper_js: _Timed | None = None
        self._grasped = False
        self._fk_cache: collections.OrderedDict[tuple[float, ...], Pose] = collections.OrderedDict()
        self._trajectories: collections.OrderedDict[str, TrajectoryArrays] = collections.OrderedDict()
        self._active_goal: Any = None
        self._gripper_goal: Any = None
        self._stop_requested = False
        self._executing = False
        self._started = False
        self._node: Any = None
        self._context: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self.availability: dict[str, bool] = {}

    @classmethod
    def from_policy(cls, policy: Any, config: Ros2BackendConfig | None = None) -> Ros2Backend:
        from armguard_mcp.policy import Policy

        assert isinstance(policy, Policy)
        r = policy.robot
        return cls(
            config or policy.ros2 or Ros2BackendConfig(),
            joint_names=r.joint_names,
            base_frame=r.base_frame,
            ee_frame=r.ee_frame,
            planning_group=r.planning_group,
            gripper_max_width_m=policy.gripper.max_width_m if policy.gripper else 0.08,
        )

    # --- rclpy <-> asyncio bridge ------------------------------------------------------
    async def _await(
        self, future: Any, timeout: float, what: str, client: Any = None, abandon: bool = True
    ) -> Any:
        """Await an rclpy Future from the asyncio loop (``call_soon_threadsafe`` bridge).

        On timeout or cancellation the rclpy future is cancelled (``abandon=True``) so late
        answers are dropped; pass ``abandon=False`` when the caller still wants to see them.
        """
        loop = asyncio.get_running_loop()
        aio: asyncio.Future[Any] = loop.create_future()

        def settle(f: Any) -> None:
            if aio.done():
                return
            if f.cancelled():
                aio.set_exception(BackendFailed(f"{what}: request was cancelled"))
                return
            exc = f.exception()
            if exc is not None:
                aio.set_exception(BackendFailed(f"{what}: {type(exc).__name__}: {exc}"))
            else:
                aio.set_result(f.result())

        def on_done(f: Any) -> None:  # runs on an executor thread (or inline if already done)
            with contextlib.suppress(RuntimeError):  # loop closed: nobody is waiting any more
                loop.call_soon_threadsafe(settle, f)

        future.add_done_callback(on_done)
        try:
            return await asyncio.wait_for(aio, timeout)
        except asyncio.TimeoutError:
            if abandon:
                self._abandon(future, client)
            raise BackendTimeout(f"{what}: no answer within {timeout:.1f} s") from None
        except BaseException:
            if abandon:
                self._abandon(future, client)
            raise

    @staticmethod
    def _abandon(future: Any, client: Any) -> None:
        if client is not None:
            with contextlib.suppress(Exception):
                client.remove_pending_request(future)
        with contextlib.suppress(Exception):
            future.cancel()

    async def _call(self, client: Any, request: Any, what: str, timeout: float | None = None) -> Any:
        if client is None:
            raise NotSupported(f"{what}: the required ROS interface package is not installed")
        if not client.service_is_ready():
            ok = await anyio.to_thread.run_sync(lambda: client.wait_for_service(timeout_sec=2.0))
            if not ok:
                raise BackendFailed(f"{what}: service {client.srv_name} is not available")
        future = client.call_async(request)
        return await self._await(future, timeout or self.cfg.service_timeout_s, what, client=client)

    async def _send_goal(
        self, action_client: Any, goal: Any, what: str, feedback: Callable[[Any], None] | None = None
    ) -> Any:
        if action_client is None:
            raise NotSupported(f"{what}: the required ROS interface package is not installed")
        if not action_client.server_is_ready():
            ok = await anyio.to_thread.run_sync(lambda: action_client.wait_for_server(timeout_sec=2.0))
            if not ok:
                name = getattr(action_client, "_action_name", "?")
                raise BackendFailed(f"{what}: action server {name} is not available")
        send_future = action_client.send_goal_async(goal, feedback_callback=feedback)
        try:
            handle = await self._await(send_future, self.cfg.service_timeout_s, what, abandon=False)
        except BaseException:
            # The goal request is already on the wire. If it is accepted late (or we were cancelled
            # while waiting), cancel it so nothing keeps moving without supervision.
            with anyio.CancelScope(shield=True), contextlib.suppress(Exception):
                late = await self._await(send_future, 2.0, f"{what} (late acceptance)")
                if late.accepted:
                    await self._cancel(late)
            raise
        if not handle.accepted:
            raise BackendFailed(f"{what}: goal rejected by the action server")
        return handle

    async def _goal_result(self, handle: Any, timeout: float, what: str) -> tuple[int, Any]:
        try:
            res = await self._await(handle.get_result_async(), timeout, what)
        except BackendTimeout:
            with contextlib.suppress(Exception):
                await self._await(handle.cancel_goal_async(), 2.0, f"{what} (cancel)")
            raise
        return res.status, res.result

    # --- lifecycle ---------------------------------------------------------------------
    async def start(self) -> None:
        if self._started:
            return
        R, cfg = self.R, self.cfg
        ctx = R.rclpy.Context()
        init_kw: dict[str, Any] = {"context": ctx, "args": []}
        if cfg.domain_id is not None:
            init_kw["domain_id"] = cfg.domain_id
        if R.SignalHandlerOptions is not None:
            init_kw["signal_handler_options"] = R.SignalHandlerOptions.NO
        R.rclpy.init(**init_kw)
        self._context = ctx
        try:
            self._create_endpoints()
        except BaseException:
            with contextlib.suppress(Exception):
                R.rclpy.try_shutdown(context=ctx)
            raise
        executor = R.MultiThreadedExecutor(num_threads=cfg.executor_threads, context=ctx)
        executor.add_node(self._node)
        self._executor = executor
        self._thread = threading.Thread(target=self._spin, name="armguard-rclpy", daemon=True)
        self._thread.start()
        self._started = True
        await self._warm_up()

    def _create_endpoints(self) -> None:
        R, cfg, ctx = self.R, self.cfg, self._context
        node = R.Node(
            cfg.node_name,
            namespace=cfg.namespace or None,
            context=ctx,
            parameter_overrides=[R.Parameter("use_sim_time", value=cfg.use_sim_time)],
        )
        self._node = node
        cb = R.ReentrantCallbackGroup()
        self._cb = cb
        sm = R.sensor_msgs

        node.create_subscription(
            sm.JointState, cfg.joint_states_topic, self._on_joint_state, 10, callback_group=cb
        )
        if cfg.wrench_topic:
            node.create_subscription(
                R.geometry_msgs.WrenchStamped,
                cfg.wrench_topic,
                self._on_wrench,
                R.qos_sensor,
                callback_group=cb,
            )
        if self.gripper_interface != "none" and cfg.gripper_joint_states_topic:
            node.create_subscription(
                sm.JointState,
                cfg.gripper_joint_states_topic,
                self._on_gripper_js,
                R.qos_sensor,
                callback_group=cb,
            )
        self._tf_buffer = self._tf_listener = None
        if R.tf2_ros is not None:
            self._tf_buffer = R.tf2_ros.Buffer(node=node)
            self._tf_listener = R.tf2_ros.TransformListener(self._tf_buffer, node, spin_thread=False)

        def client(pkg: Any, srv: str, name: str | None) -> Any:
            if pkg is None or not name:
                return None
            return node.create_client(getattr(pkg, srv), name, callback_group=cb)

        def action(pkg: Any, act: str, name: str | None) -> Any:
            if pkg is None or not name:
                return None
            return R.ActionClient(node, getattr(pkg, act), name, callback_group=cb)

        cm = cfg.controller_manager.rstrip("/")
        self._fk_client = (
            client(R.moveit_srv, "GetPositionFK", cfg.compute_fk_service) if self._kin is None else None
        )
        self._plan_client = client(R.moveit_srv, "GetMotionPlan", cfg.plan_service)
        self._cartesian_client = client(R.moveit_srv, "GetCartesianPath", cfg.cartesian_path_service)
        self._list_ctrl_client = client(R.cm_srv, "ListControllers", f"{cm}/list_controllers")
        self._switch_ctrl_client = client(R.cm_srv, "SwitchController", f"{cm}/switch_controller")
        self._fjt_client = action(R.control_action, "FollowJointTrajectory", cfg.trajectory_action)
        self._collision_client = client(
            R.franka_srv, "SetForceTorqueCollisionBehavior", cfg.collision_behavior_service
        )
        self._recovery_client = action(R.franka_action, "ErrorRecovery", cfg.error_recovery_action)
        gns = cfg.gripper_namespace.rstrip("/")
        self._gripper_clients: dict[str, Any] = {}
        if self.gripper_interface == "franka":
            for act in ("Move", "Grasp", "Homing"):
                self._gripper_clients[act] = action(R.franka_action, act, f"{gns}/{act.lower()}")
        elif self.gripper_interface == "gripper_command":
            self._gripper_clients["GripperCommand"] = action(
                R.control_action, "GripperCommand", cfg.gripper_command_action
            )

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as e:  # ExternalShutdownException / ShutdownException on shutdown
            if self._started:
                logger.error("rclpy executor stopped: %s: %s", type(e).__name__, e)

    async def _warm_up(self) -> None:
        """DDS discovery: wait (in parallel, bounded) for the endpoints we will need, and log what is missing."""
        timeout = self.cfg.startup_timeout_s
        checks: dict[str, Callable[[], bool]] = {}
        for label, c in (
            ("compute_fk", self._fk_client),
            ("plan_kinematic_path", self._plan_client),
            ("compute_cartesian_path", self._cartesian_client),
            ("list_controllers", self._list_ctrl_client),
            ("switch_controller", self._switch_ctrl_client),
            ("collision_behavior", self._collision_client),
        ):
            if c is not None:
                checks[label] = lambda c=c: c.wait_for_service(timeout_sec=timeout)
        for label, a in (
            ("follow_joint_trajectory", self._fjt_client),
            ("error_recovery", self._recovery_client),
        ):
            if a is not None:
                checks[label] = lambda a=a: a.wait_for_server(timeout_sec=timeout)
        for act, a in self._gripper_clients.items():
            if a is not None:
                checks[f"gripper_{act.lower()}"] = lambda a=a: a.wait_for_server(timeout_sec=timeout)

        results: dict[str, bool] = {}

        async def run(label: str, fn: Callable[[], bool]) -> None:
            results[label] = bool(await anyio.to_thread.run_sync(fn))

        async def first_joint_state() -> None:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not self._have_all_joints():
                await anyio.sleep(0.02)
            results["joint_states"] = self._have_all_joints()

        async def first_tf() -> None:
            buf, zero = self._tf_buffer, self.R.Time()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not buf.can_transform(self.base_frame, self.ee_frame, zero):
                await anyio.sleep(0.02)
            results["tf"] = buf.can_transform(self.base_frame, self.ee_frame, zero)

        async with anyio.create_task_group() as tg:
            for label, fn in checks.items():
                tg.start_soon(run, label, fn)
            tg.start_soon(first_joint_state)
            if self._tf_buffer is not None:
                tg.start_soon(first_tf)
        self.availability = results
        missing = sorted(k for k, ok in results.items() if not ok)
        if missing:
            logger.warning("ros2 backend: not available after %.1f s: %s", timeout, ", ".join(missing))
        else:
            logger.info("ros2 backend: all %d endpoints available", len(results))

    async def shutdown(self) -> None:
        """Best-effort, loop-independent teardown (the CLI calls it after the server loop has ended)."""
        if not self._started:
            return
        for handle in (self._active_goal, self._gripper_goal):
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.cancel_goal_async()
        self._started = False
        with contextlib.suppress(Exception):
            self._executor.shutdown(timeout_sec=2.0)
        with contextlib.suppress(Exception):
            self._node.destroy_node()
        with contextlib.suppress(Exception):
            self.R.rclpy.try_shutdown(context=self._context)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # --- subscription callbacks (executor threads) --------------------------------------
    def _on_joint_state(self, msg: Any) -> None:
        now = time.monotonic()
        n = len(msg.name)
        pos, vel, eff = msg.position, msg.velocity, msg.effort
        stamp = duration_to_s(msg.header.stamp)
        with self._lock:
            for i, name in enumerate(msg.name):
                if i >= len(pos):
                    break
                self._joints[name] = (
                    float(pos[i]),
                    float(vel[i]) if len(vel) == n else 0.0,
                    float(eff[i]) if len(eff) == n else 0.0,
                    now,
                )
            self._joint_stamp = stamp

    def _on_wrench(self, msg: Any) -> None:
        self._wrench = _Timed(msg, time.monotonic())

    def _on_gripper_js(self, msg: Any) -> None:
        self._gripper_js = _Timed(msg, time.monotonic())

    # --- state -------------------------------------------------------------------------
    def _require_started(self) -> None:
        if not self._started:
            raise BackendFailed("ros2 backend is not started")

    def _have_all_joints(self) -> bool:
        with self._lock:
            return all(j in self._joints for j in self.joint_names)

    def _joint_snapshot(self) -> tuple[list[tuple[float, float, float, float]] | None, list[str], float]:
        with self._lock:
            missing = [j for j in self.joint_names if j not in self._joints]
            if missing:
                return None, missing, 0.0
            return [self._joints[j] for j in self.joint_names], [], self._joint_stamp

    async def get_joint_state(self) -> JointState:
        self._require_started()
        max_age = self.cfg.joint_state_max_age_s
        deadline = time.monotonic() + max(max_age, 1.0)
        while True:
            entries, missing, stamp = self._joint_snapshot()
            now = time.monotonic()
            age = max(now - e[3] for e in entries) if entries else math.inf
            if entries is not None and age <= max_age:
                return JointState(
                    names=list(self.joint_names),
                    positions=[e[0] for e in entries],
                    velocities=[e[1] for e in entries],
                    efforts=[e[2] for e in entries],
                    stamp=stamp or time.time(),
                )
            if now >= deadline:
                if entries is None:
                    raise BackendTimeout(
                        f"no joint state for {missing} on {self.cfg.joint_states_topic} (is the "
                        "joint_state_broadcaster running?)"
                    )
                raise BackendTimeout(
                    f"joint states on {self.cfg.joint_states_topic} are stale (newest joint {age:.2f} s old, "
                    f"max {max_age} s)"
                )
            await anyio.sleep(0.01)

    def _pose_from_transform(self, t: Any, frame: str) -> Pose:
        tr, rot = t.transform.translation, t.transform.rotation
        return Pose(
            frame_id=frame,
            position=Vector3(x=tr.x, y=tr.y, z=tr.z),
            orientation=Quaternion(x=rot.x, y=rot.y, z=rot.z, w=rot.w),
        )

    async def get_ee_pose(self) -> Pose:
        if self._tf_buffer is None:  # no tf2_ros: FK of the current joint state
            return await self.forward_kinematics((await self.get_joint_state()).positions)
        return await self.lookup_transform(self.base_frame, self.ee_frame)

    async def lookup_transform(self, target_frame: str, source_frame: str) -> Pose:
        self._require_started()
        if self._tf_buffer is None:
            raise NotSupported("tf2_ros is not installed")
        R = self.R
        zero = R.Time()
        buf = self._tf_buffer
        if not buf.can_transform(target_frame, source_frame, zero):
            try:
                await self._await(
                    buf.wait_for_transform_async(target_frame, source_frame, zero),
                    self.cfg.tf_timeout_s,
                    f"tf {target_frame} <- {source_frame}",
                )
            except BackendTimeout:
                try:
                    frames = sorted(buf._getFrameStrings())
                except Exception:
                    frames = []
                raise BackendFailed(
                    f"no transform {target_frame} <- {source_frame} within {self.cfg.tf_timeout_s} s"
                    + (f"; known frames: {frames[:40]}" if frames else "")
                ) from None
        try:
            t = buf.lookup_transform(target_frame, source_frame, zero)
        except Exception as e:  # tf2 LookupException / ExtrapolationException / ...
            raise BackendFailed(f"tf lookup {target_frame} <- {source_frame} failed: {e}") from e
        return self._pose_from_transform(t, target_frame)

    async def get_wrench(self) -> Wrench | None:
        cfg = self.cfg
        if not cfg.wrench_topic:
            return None
        self._require_started()
        deadline = time.monotonic() + cfg.wrench_max_age_s
        while True:
            w = self._wrench
            now = time.monotonic()
            if w is not None and now - w.received <= cfg.wrench_max_age_s:
                m = w.value
                f, t = m.wrench.force, m.wrench.torque
                return Wrench(
                    frame_id=m.header.frame_id or self.base_frame,
                    force=Vector3(x=f.x, y=f.y, z=f.z),
                    torque=Vector3(x=t.x, y=t.y, z=t.z),
                    stamp=duration_to_s(m.header.stamp) or time.time(),
                )
            if now >= deadline:
                if not cfg.require_wrench:
                    return None
                what = "no message yet" if w is None else f"newest is {now - w.received:.2f} s old"
                raise BackendTimeout(
                    f"no fresh external wrench on {cfg.wrench_topic} ({what}); force limits cannot be "
                    "monitored, so motion is refused (set ros2.require_wrench: false to allow unmonitored motion)"
                )
            await anyio.sleep(0.005)

    async def forward_kinematics(self, joint_positions: Sequence[float]) -> Pose:
        q = [float(v) for v in joint_positions]
        if len(q) != len(self.joint_names):
            raise BackendFailed(
                f"forward_kinematics needs {len(self.joint_names)} joint positions, got {len(q)}"
            )
        key = tuple(round(v, 9) for v in q)
        cached = self._fk_cache.get(key)
        if cached is not None:
            self._fk_cache.move_to_end(key)
            return cached
        if self._kin is not None:
            t = self._kin.chain(q)[self.ee_frame]
            pose = Pose(
                frame_id=self.base_frame,
                position=Vector3.of(g.translation(t)),
                orientation=Quaternion.of(g.quat_from_matrix(t)),
            )
        else:
            self._require_started()
            R = self.R
            if R.moveit_srv is None:
                raise NotSupported("moveit_msgs is not installed (needed for /compute_fk)")
            req = R.moveit_srv.GetPositionFK.Request()
            req.header.frame_id = self.base_frame
            req.fk_link_names = [self.ee_frame]
            req.robot_state = self._robot_state_msg(q)
            res = await self._call(self._fk_client, req, "forward kinematics (/compute_fk)")
            if res.error_code.val != R.moveit_msg.MoveItErrorCodes.SUCCESS or not res.pose_stamped:
                raise BackendFailed(
                    f"/compute_fk failed: {_moveit_code_name(R.moveit_msg, res.error_code.val)}"
                )
            ps = res.pose_stamped[0]
            if ps.header.frame_id.lstrip("/") != self.base_frame:
                raise BackendFailed(
                    f"/compute_fk answered in frame {ps.header.frame_id!r}, not {self.base_frame!r}"
                )
            p, o = ps.pose.position, ps.pose.orientation
            pose = Pose(
                frame_id=self.base_frame,
                position=Vector3(x=p.x, y=p.y, z=p.z),
                orientation=Quaternion(x=o.x, y=o.y, z=o.z, w=o.w),
            )
        self._fk_cache[key] = pose
        if len(self._fk_cache) > 4096:
            self._fk_cache.popitem(last=False)
        return pose

    def _robot_state_msg(self, q: Sequence[float]) -> Any:
        rs = self.R.moveit_msg.RobotState()
        rs.joint_state.name = list(self.joint_names)
        rs.joint_state.position = [float(v) for v in q]
        rs.is_diff = True  # other joints (e.g. fingers) keep move_group's current values
        return rs

    # --- controllers -------------------------------------------------------------------
    async def list_controllers(self) -> list[ControllerInfo]:
        self._require_started()
        if self.R.cm_srv is None:
            raise NotSupported("controller_manager_msgs is not installed")
        res = await self._call(
            self._list_ctrl_client, self.R.cm_srv.ListControllers.Request(), "list_controllers"
        )
        return [ControllerInfo(name=c.name, type=c.type, state=c.state) for c in res.controller]

    async def switch_controllers(
        self, activate: Sequence[str], deactivate: Sequence[str]
    ) -> list[ControllerInfo]:
        self._require_started()
        if self.R.cm_srv is None:
            raise NotSupported("controller_manager_msgs is not installed")
        if self._executing:
            raise BackendFailed("cannot switch controllers while a trajectory is executing; stop it first")
        srv = self.R.cm_srv.SwitchController
        req = srv.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        req.strictness = (
            srv.Request.STRICT if self.cfg.switch_strictness == "strict" else srv.Request.BEST_EFFORT
        )
        req.activate_asap = True
        req.timeout.sec, req.timeout.nanosec = s_to_sec_nanosec(self.cfg.switch_timeout_s)
        res = await self._call(
            self._switch_ctrl_client, req, "switch_controller", timeout=self.cfg.switch_timeout_s + 5.0
        )
        if not res.ok:
            detail = getattr(res, "message", "") or "see the controller_manager log"
            raise BackendFailed(f"controller switch refused by controller_manager: {detail}")
        return await self.list_controllers()

    # --- planning ----------------------------------------------------------------------
    def _make_plan(
        self, kind: PlanKind, arrays: TrajectoryArrays, vs: float, as_: float, source: str
    ) -> Plan:
        plan = Plan(
            plan_id=new_plan_id(),
            kind=kind,
            joint_names=list(self.joint_names),
            waypoints=arrays.waypoints,
            time_from_start=arrays.times,
            duration_s=arrays.times[-1],
            created_at=time.time(),
            source_request=source,
            velocity_scaling=vs,
            acceleration_scaling=as_,
        )
        self._trajectories[plan.plan_id] = arrays
        while len(self._trajectories) > _MAX_STORED_TRAJECTORIES:
            self._trajectories.popitem(last=False)
        return plan

    def _base_request(self, vs: float, as_: float) -> Any:
        R, cfg = self.R, self.cfg
        if R.moveit_msg is None:
            raise NotSupported("moveit_msgs is not installed: planning needs MoveIt 2's move_group")
        req = R.moveit_msg.MotionPlanRequest()
        req.group_name = self.planning_group
        req.pipeline_id = cfg.pipeline_id
        req.planner_id = cfg.planner_id
        req.num_planning_attempts = cfg.planning_attempts
        req.allowed_planning_time = cfg.planning_time_s
        req.max_velocity_scaling_factor = float(vs)
        req.max_acceleration_scaling_factor = float(as_)
        return req

    async def _plan(self, req: Any, kind: PlanKind, vs: float, as_: float, source: str) -> Plan:
        R = self.R
        req.start_state = self._robot_state_msg((await self.get_joint_state()).positions)
        srv_req = R.moveit_srv.GetMotionPlan.Request()
        srv_req.motion_plan_request = req
        res = await self._call(
            self._plan_client,
            srv_req,
            "motion planning (/plan_kinematic_path)",
            timeout=self.cfg.planning_time_s + 5.0,
        )
        mpr = res.motion_plan_response
        if mpr.error_code.val != R.moveit_msg.MoveItErrorCodes.SUCCESS:
            raise BackendFailed(
                f"MoveIt planning failed: {_moveit_code_name(R.moveit_msg, mpr.error_code.val)}"
            )
        jt = mpr.trajectory.joint_trajectory
        arrays = trajectory_arrays(jt.joint_names, jt.points, self.joint_names)
        return self._make_plan(kind, arrays, vs, as_, source)

    async def plan_to_joints(self, target: Sequence[float], vel_scale: float, acc_scale: float) -> Plan:
        self._require_started()
        if len(target) != len(self.joint_names):
            raise BackendFailed(f"expected {len(self.joint_names)} joint positions, got {len(target)}")
        req = self._base_request(vel_scale, acc_scale)
        c = self.R.moveit_msg.Constraints()
        tol = self.cfg.goal_joint_tolerance_rad
        for name, q in zip(self.joint_names, target, strict=True):
            jc = self.R.moveit_msg.JointConstraint()
            jc.joint_name, jc.position = name, float(q)
            jc.tolerance_above = jc.tolerance_below = tol
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]
        return await self._plan(
            req, "joints", vel_scale, acc_scale, f"joints -> {[round(float(v), 4) for v in target]}"
        )

    def _check_base_frame(self, pose: Pose) -> None:
        if pose.frame_id and pose.frame_id != self.base_frame:
            raise BackendFailed(f"pose must be in {self.base_frame}, got {pose.frame_id!r}")

    async def plan_to_pose(self, pose: Pose, vel_scale: float, acc_scale: float) -> Plan:
        self._require_started()
        self._check_base_frame(pose)
        R, cfg = self.R, self.cfg
        req = self._base_request(vel_scale, acc_scale)
        if R.shape_msgs is None:
            raise NotSupported("shape_msgs is not installed")
        pc = R.moveit_msg.PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_frame
        sphere = R.shape_msgs.SolidPrimitive()
        sphere.type = R.shape_msgs.SolidPrimitive.SPHERE
        sphere.dimensions = [cfg.goal_position_tolerance_m]
        region_pose = R.geometry_msgs.Pose()
        region_pose.position.x, region_pose.position.y, region_pose.position.z = pose.position.as_tuple()
        region_pose.orientation.w = 1.0
        pc.constraint_region.primitives = [sphere]
        pc.constraint_region.primitive_poses = [region_pose]
        pc.weight = 1.0
        oc = R.moveit_msg.OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.ee_frame
        o = pose.orientation
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = o.x, o.y, o.z, o.w
        tol = cfg.goal_orientation_tolerance_rad
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = tol
        oc.weight = 1.0
        c = R.moveit_msg.Constraints()
        c.position_constraints = [pc]
        c.orientation_constraints = [oc]
        req.goal_constraints = [c]
        p = pose.position
        return await self._plan(
            req, "pose", vel_scale, acc_scale, f"pose -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )

    async def plan_cartesian(
        self, waypoints: Sequence[Pose], max_step: float, vel_scale: float, acc_scale: float
    ) -> Plan:
        self._require_started()
        R, cfg = self.R, self.cfg
        if R.moveit_srv is None:
            raise NotSupported("moveit_msgs is not installed: Cartesian planning needs MoveIt 2's move_group")
        if not waypoints:
            raise BackendFailed("plan_cartesian needs at least one waypoint")
        if max_step <= 0:
            raise BackendFailed("max_step must be > 0")
        req = R.moveit_srv.GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.start_state = self._robot_state_msg((await self.get_joint_state()).positions)
        req.group_name = self.planning_group
        req.link_name = self.ee_frame
        msgs = []
        for wp in waypoints:
            self._check_base_frame(wp)
            m = R.geometry_msgs.Pose()
            m.position.x, m.position.y, m.position.z = wp.position.as_tuple()
            o = wp.orientation
            m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = o.x, o.y, o.z, o.w
            msgs.append(m)
        req.waypoints = msgs
        req.max_step = float(max_step)
        req.jump_threshold = cfg.cartesian_jump_threshold
        # Fields added in newer moveit_msgs (present on Jazzy 2.6); tolerate older definitions.
        for field_name, value in (
            ("revolute_jump_threshold", cfg.cartesian_revolute_jump_threshold_rad),
            ("prismatic_jump_threshold", 0.0),
            ("max_velocity_scaling_factor", float(vel_scale)),
            ("max_acceleration_scaling_factor", float(acc_scale)),
        ):
            if hasattr(req, field_name):
                setattr(req, field_name, value)
        req.avoid_collisions = cfg.cartesian_avoid_collisions
        res = await self._call(
            self._cartesian_client,
            req,
            "Cartesian planning (/compute_cartesian_path)",
            timeout=cfg.planning_time_s + 5.0,
        )
        if res.error_code.val != R.moveit_msg.MoveItErrorCodes.SUCCESS:
            raise BackendFailed(
                f"Cartesian planning failed: {_moveit_code_name(R.moveit_msg, res.error_code.val)}"
            )
        if res.fraction < cfg.cartesian_min_fraction - 1e-9:
            raise BackendFailed(
                f"Cartesian path not feasible: only {100 * res.fraction:.1f}% of the path could be followed "
                "(IK failure, joint jump or collision)"
            )
        jt = res.solution.joint_trajectory
        arrays = trajectory_arrays(jt.joint_names, jt.points, self.joint_names)
        return self._make_plan(
            "cartesian", arrays, vel_scale, acc_scale, f"cartesian, {len(waypoints)} waypoint(s)"
        )

    # --- execution ---------------------------------------------------------------------
    def _trajectory_goal(self, plan: Plan) -> Any:
        R = self.R
        goal = R.control_action.FollowJointTrajectory.Goal()
        jt = goal.trajectory
        jt.joint_names = list(plan.joint_names)
        extras = self._trajectories.pop(plan.plan_id, None)
        if extras is not None and (
            extras.waypoints != plan.waypoints or extras.times != plan.time_from_start
        ):
            extras = None  # never attach derivatives that belong to other points
        vels = extras.velocities if extras is not None else None
        accs = extras.accelerations if extras is not None and vels is not None else None
        points = []
        for k, (q, t) in enumerate(zip(plan.waypoints, plan.time_from_start, strict=True)):
            pt = R.trajectory_msgs.JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            if vels is not None:
                pt.velocities = vels[k]
            if accs is not None:
                pt.accelerations = accs[k]
            pt.time_from_start.sec, pt.time_from_start.nanosec = s_to_sec_nanosec(t)
            points.append(pt)
        jt.points = points
        goal.goal_time_tolerance.sec, goal.goal_time_tolerance.nanosec = s_to_sec_nanosec(
            self.cfg.goal_time_tolerance_s
        )
        return goal

    def _current_force(self) -> float:
        w = self._wrench
        if w is None:
            return 0.0
        f = w.value.wrench.force
        return math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)

    async def execute(
        self, plan: Plan, on_progress: ProgressCallback, should_abort: AbortCheck
    ) -> ExecutionResult:
        self._require_started()
        R, cfg = self.R, self.cfg
        if R.control_action is None:
            raise NotSupported("control_msgs is not installed (needed for FollowJointTrajectory)")
        if self._executing:
            raise BackendFailed("another trajectory is already executing")
        if plan.joint_names != self.joint_names:
            raise BackendFailed("plan joint names do not match the robot")
        js = await self.get_joint_state()
        dev = max(abs(a - b) for a, b in zip(plan.start, js.positions, strict=True))
        if dev > cfg.start_tolerance_rad:
            raise BackendFailed(f"robot is {dev:.4f} rad away from the plan's start state")

        goal = self._trajectory_goal(plan)
        loop = asyncio.get_running_loop()
        latest: dict[str, float] = {}

        def on_feedback(fb_msg: Any) -> None:  # executor thread
            t = duration_to_s(fb_msg.feedback.desired.time_from_start)
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(latest.__setitem__, "t", t)

        self._executing = True
        self._stop_requested = False
        max_force = self._current_force()
        handle: Any = None
        result_future: Any = None
        try:
            handle = await self._send_goal(
                self._fjt_client, goal, f"trajectory execution ({cfg.trajectory_action})", on_feedback
            )
            self._active_goal = handle
            result_future = handle.get_result_async()
            t0 = time.monotonic()
            deadline = t0 + plan.duration_s + cfg.goal_time_tolerance_s + cfg.execution_timeout_margin_s
            next_report = 0.0
            aborted = False
            while not result_future.done():
                max_force = max(max_force, self._current_force())
                if self._stop_requested or should_abort():
                    aborted = True
                    break
                if time.monotonic() > deadline:
                    await self._cancel(handle)
                    raise BackendTimeout(
                        f"controller did not finish the {plan.duration_s:.2f} s trajectory in time; goal cancelled"
                    )
                t = latest.get("t", time.monotonic() - t0)
                frac = min(0.99, t / plan.duration_s) if plan.duration_s > 0 else 0.99
                if frac >= next_report:
                    await on_progress(frac, f"executing {plan.plan_id}: {100 * frac:.0f}%")
                    next_report = frac + 0.1
                await anyio.sleep(cfg.poll_period_s)
            if aborted:
                await self._cancel(handle)
                status, _ = await self._wait_result(result_future, 3.0)
                final = await self._settled_joint_state()
                return ExecutionResult(
                    success=False,
                    message=f"aborted: trajectory goal cancelled (controller status {_STATUS_NAMES.get(status, status)})",
                    final_joint_state=final,
                    max_observed_force_n=max_force,
                    aborted=True,
                )
            res = result_future.result()
            status, result = res.status, res.result
            max_force = max(max_force, self._current_force())
            final = await self._settled_joint_state()
            ok = status == _STATUS_SUCCEEDED and result.error_code == 0
            if ok:
                await on_progress(1.0, "trajectory complete")
                msg = "trajectory complete"
            else:
                msg = (
                    f"controller reported {_STATUS_NAMES.get(status, status)} (error_code {result.error_code}"
                    + (f": {result.error_string}" if result.error_string else "")
                    + ")"
                )
            return ExecutionResult(
                success=ok,
                message=msg,
                final_joint_state=final,
                max_observed_force_n=max_force,
                aborted=status == _STATUS_CANCELED,
            )
        except BaseException:
            # Fail-safe: whatever interrupted us (task cancellation, client disconnect, error),
            # never leave a trajectory running unsupervised.
            if handle is not None and (result_future is None or not result_future.done()):
                with anyio.CancelScope(shield=True):
                    await self._cancel(handle)
            raise
        finally:
            self._active_goal = None
            self._executing = False

    async def _wait_result(self, result_future: Any, timeout: float) -> tuple[int, Any]:
        try:
            res = await self._await(result_future, timeout, "trajectory result after cancel")
        except Exception:
            return 0, None
        return res.status, res.result

    async def _cancel(self, handle: Any) -> None:
        try:
            await self._await(handle.cancel_goal_async(), 2.0, "cancel trajectory goal")
        except Exception as e:  # never raise from a stop path
            logger.warning("cancelling the trajectory goal failed: %s", e)

    async def _settled_joint_state(self) -> JointState | None:
        """Joint state received after the motion ended (so final positions are not a stale sample)."""
        t_end = time.monotonic()
        deadline = t_end + 0.5
        while time.monotonic() < deadline:
            entries, _, _ = self._joint_snapshot()
            if entries is not None and min(e[3] for e in entries) > t_end:
                break
            await anyio.sleep(0.01)
        try:
            return await self.get_joint_state()
        except Exception:
            return None

    async def stop(self) -> None:
        self._stop_requested = True
        if not self._started:
            return
        for handle in (self._active_goal, self._gripper_goal):
            if handle is not None:
                await self._cancel(handle)

    # --- gripper -----------------------------------------------------------------------
    def _gripper_width(self) -> float | None:
        js = self._gripper_js
        if js is None or not js.value.position:
            return None
        return self.cfg.gripper_width_scale * float(sum(js.value.position))

    def _gripper_state(self, fallback_width: float | None = None) -> GripperState:
        width = self._gripper_width()
        if width is None:  # no gripper joint_states: report the commanded width
            width = fallback_width if fallback_width is not None else 0.0
        return GripperState(
            width_m=width, max_width_m=self.gripper_max_width, is_grasped=self._grasped, stamp=time.time()
        )

    async def _gripper_goal_run(self, kind: str, goal: Any, what: str) -> tuple[int, Any]:
        client = self._gripper_clients.get(kind)
        if self.gripper_interface == "none" or client is None:
            raise NotSupported("no gripper interface is configured (ros2.gripper_interface)")
        handle = await self._send_goal(client, goal, what)
        self._gripper_goal = handle
        try:
            return await self._goal_result(handle, self.cfg.gripper_timeout_s, what)
        finally:
            self._gripper_goal = None

    async def _gripper_settle(self) -> None:
        # give the gripper's joint_states a moment to reflect the final width
        await anyio.sleep(0.05)

    async def gripper_move(self, width: float, speed: float) -> GripperState:
        self._require_started()
        A = self.R.franka_action
        if self.gripper_interface == "franka":
            goal = A.Move.Goal(width=float(width), speed=float(speed))
            status, res = await self._gripper_goal_run("Move", goal, "gripper move")
            if status != _STATUS_SUCCEEDED or not res.success:
                raise BackendFailed(
                    f"gripper move failed ({_STATUS_NAMES.get(status, status)}): {res.error or 'no detail'}"
                )
        else:
            status, res = await self._gripper_command(width, 0.0, "gripper move")
            if status != _STATUS_SUCCEEDED:
                raise BackendFailed(f"gripper move failed ({_STATUS_NAMES.get(status, status)})")
        self._grasped = False
        await self._gripper_settle()
        return self._gripper_state(width)

    async def _gripper_command(self, width: float, effort: float, what: str) -> tuple[int, Any]:
        goal = self.R.control_action.GripperCommand.Goal()
        pos = width / 2.0 if self.cfg.gripper_command_position_is_half_width else width
        goal.command.position = float(pos)
        goal.command.max_effort = float(effort)
        return await self._gripper_goal_run("GripperCommand", goal, what)

    async def gripper_grasp(
        self, width: float, force: float, speed: float, epsilon_inner: float, epsilon_outer: float
    ) -> GripperState:
        self._require_started()
        if self.gripper_interface == "franka":
            A, M = self.R.franka_action, self.R.franka_msg
            goal = A.Grasp.Goal()
            goal.width, goal.speed, goal.force = float(width), float(speed), float(force)
            goal.epsilon = M.GraspEpsilon(inner=float(epsilon_inner), outer=float(epsilon_outer))
            status, res = await self._gripper_goal_run("Grasp", goal, "gripper grasp")
            if status not in (_STATUS_SUCCEEDED, _STATUS_ABORTED):
                raise BackendFailed(f"gripper grasp failed ({_STATUS_NAMES.get(status, status)})")
            if not res.success and res.error:  # libfranka exception, not merely "no object"
                raise BackendFailed(f"gripper grasp failed: {res.error}")
            self._grasped = bool(res.success)
        else:
            # GripperCommand has no grasp semantics: aim epsilon_inner inside the object width so the
            # fingers squeeze and stall on it (like franka's Grasp), instead of stopping just at contact.
            target = max(0.0, width - epsilon_inner)
            status, res = await self._gripper_command(target, force, "gripper grasp")
            if status not in (_STATUS_SUCCEEDED, _STATUS_ABORTED):
                raise BackendFailed(f"gripper grasp failed ({_STATUS_NAMES.get(status, status)})")
            self._grasped = bool(res.stalled) or (status == _STATUS_ABORTED and res.position > 0.0)
        await self._gripper_settle()
        return self._gripper_state(width)

    async def gripper_home(self) -> GripperState:
        self._require_started()
        if self.gripper_interface == "franka":
            status, res = await self._gripper_goal_run(
                "Homing", self.R.franka_action.Homing.Goal(), "gripper homing"
            )
            if status != _STATUS_SUCCEEDED or not res.success:
                raise BackendFailed(
                    f"gripper homing failed ({_STATUS_NAMES.get(status, status)}): {res.error or 'no detail'}"
                )
            self._grasped = False
            await self._gripper_settle()
            return self._gripper_state(self.gripper_max_width)
        return await self.gripper_move(self.gripper_max_width, 0.1)

    async def get_gripper_state(self) -> GripperState | None:
        if self.gripper_interface == "none" or not self._started:
            return None
        if self._gripper_width() is None:
            return None
        return self._gripper_state()

    # --- perception --------------------------------------------------------------------
    async def camera_snapshot(self, topic: str, max_width: int) -> CameraFrame:
        self._require_started()
        R, node = self.R, self._node
        types = dict(node.get_topic_names_and_types()).get(topic, [])
        if "sensor_msgs/msg/CompressedImage" in types:
            msg_type = R.sensor_msgs.CompressedImage
        elif "sensor_msgs/msg/Image" in types:
            msg_type = R.sensor_msgs.Image
        elif types:
            raise NotSupported(
                f"{topic} has type {types}; only sensor_msgs Image/CompressedImage are supported"
            )
        else:
            raise BackendFailed(f"no publisher on camera topic {topic}")
        loop = asyncio.get_running_loop()
        got: asyncio.Future[Any] = loop.create_future()

        def on_msg(m: Any) -> None:
            def settle() -> None:
                if not got.done():
                    got.set_result(m)

            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(settle)

        sub = node.create_subscription(msg_type, topic, on_msg, R.qos_sensor, callback_group=self._cb)
        try:
            msg = await asyncio.wait_for(got, self.cfg.camera_timeout_s)
        except asyncio.TimeoutError:
            raise BackendTimeout(f"no image on {topic} within {self.cfg.camera_timeout_s} s") from None
        finally:
            node.destroy_subscription(sub)
        stamp = duration_to_s(msg.header.stamp) or time.time()
        data = bytes(msg.data)
        if msg_type is R.sensor_msgs.CompressedImage:
            mime, w, h = compressed_image_info(data)
            return CameraFrame(mime, data, w, h, stamp)
        png, w, h = await anyio.to_thread.run_sync(
            raw_image_to_png, msg.encoding, msg.width, msg.height, msg.step, data, max_width
        )
        return CameraFrame("image/png", png, w, h, stamp)

    # --- robot-specific ----------------------------------------------------------------
    async def set_collision_thresholds(self, force_n: float, torque_nm: float) -> None:
        self._require_started()
        if self.R.franka_srv is None or self._collision_client is None:
            raise NotSupported(
                "collision thresholds need franka_msgs and ros2.collision_behavior_service (franka_ros2)"
            )
        if self._executing:
            raise BackendFailed("cannot change collision thresholds while a trajectory is executing")
        req = self.R.franka_srv.SetForceTorqueCollisionBehavior.Request()
        joints = [float(v) for v in self.cfg.collision_joint_torque_thresholds_nm]
        cart = [float(force_n)] * 3 + [float(torque_nm)] * 3
        req.lower_torque_thresholds_nominal = joints
        req.upper_torque_thresholds_nominal = joints
        req.lower_force_thresholds_nominal = cart
        req.upper_force_thresholds_nominal = cart
        res = await self._call(self._collision_client, req, "set_force_torque_collision_behavior")
        if not res.success:
            raise BackendFailed(f"robot refused the collision thresholds: {res.error or 'no detail'}")

    async def error_recovery(self) -> None:
        self._require_started()
        if self.R.franka_action is None or self._recovery_client is None:
            raise NotSupported(
                "error recovery needs franka_msgs and ros2.error_recovery_action (franka_ros2)"
            )
        if self._executing:
            raise NotSupported("error recovery is not possible while a trajectory is executing")
        handle = await self._send_goal(
            self._recovery_client, self.R.franka_action.ErrorRecovery.Goal(), "error recovery"
        )
        status, _ = await self._goal_result(handle, self.cfg.error_recovery_timeout_s, "error recovery")
        if status != _STATUS_SUCCEEDED:
            raise BackendFailed(
                f"error recovery failed ({_STATUS_NAMES.get(status, status)}); check the robot's Desk UI"
            )

    async def list_graph(self) -> GraphInfo:
        self._require_started()
        node = self._node
        nodes = [f"{ns.rstrip('/')}/{n}" for n, ns in node.get_node_names_and_namespaces()]
        topics = [t for t, _ in node.get_topic_names_and_types()]
        services = [s for s, _ in node.get_service_names_and_types()]
        from rclpy.action import get_action_names_and_types

        actions = [a for a, _ in get_action_names_and_types(node)]
        own = f"{node.get_namespace().rstrip('/')}/{node.get_name()}"
        return filter_graph(nodes, topics, services, actions, own)
