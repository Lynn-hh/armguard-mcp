"""The robot backend interface.

A backend talks to the robot (or a simulation). It knows nothing about MCP, policies or
approvals: the server enforces the safety envelope *before* calling a backend and monitors
execution while it runs. Backends must be safe to call from a single asyncio event loop.
"""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable, Sequence
from typing import NamedTuple

from armguard_mcp.models import (
    ControllerInfo,
    ExecutionResult,
    GraphInfo,
    GripperState,
    JointState,
    Plan,
    Pose,
    Wrench,
)

ProgressCallback = Callable[[float, str], Awaitable[None]]
"""``await on_progress(fraction_0_to_1, message)``"""

AbortCheck = Callable[[], bool]
"""Polled by ``execute`` every control tick; returning True must stop the motion promptly."""


class BackendError(Exception):
    """Base class for backend failures. Messages are shown to the LLM, so keep them informative."""


class NotSupported(BackendError):
    pass


class BackendTimeout(BackendError):
    pass


class BackendFailed(BackendError):
    pass


# Aliases matching the names used in the design notes.
Timeout = BackendTimeout
Failed = BackendFailed


class CameraFrame(NamedTuple):
    mime: str
    data: bytes
    width: int
    height: int
    stamp: float


class RobotBackend(abc.ABC):
    """Abstract robot backend. All methods are coroutines."""

    name: str = "abstract"

    # --- lifecycle ---------------------------------------------------------------------
    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def shutdown(self) -> None: ...

    # --- state -------------------------------------------------------------------------
    @abc.abstractmethod
    async def get_joint_state(self) -> JointState: ...

    @abc.abstractmethod
    async def get_ee_pose(self) -> Pose: ...

    @abc.abstractmethod
    async def forward_kinematics(self, joint_positions: Sequence[float]) -> Pose:
        """TCP pose in the robot base frame for the given joint positions."""

    @abc.abstractmethod
    async def lookup_transform(self, target_frame: str, source_frame: str) -> Pose:
        """Pose of ``source_frame`` expressed in ``target_frame`` (tf2 semantics)."""

    @abc.abstractmethod
    async def get_wrench(self) -> Wrench | None:
        """Estimated external wrench at the TCP, or None if the robot has no estimate.

        Polled by the server's force monitor while a plan executes. Return promptly (the server
        treats a read slower than ``force.wrench_timeout_s`` as a monitor failure) and set
        ``stamp`` to the time of the measurement, so a wrench that stops updating can be detected.
        """

    # --- controllers -------------------------------------------------------------------
    @abc.abstractmethod
    async def list_controllers(self) -> list[ControllerInfo]: ...

    @abc.abstractmethod
    async def switch_controllers(
        self, activate: Sequence[str], deactivate: Sequence[str]
    ) -> list[ControllerInfo]: ...

    # --- planning ----------------------------------------------------------------------
    @abc.abstractmethod
    async def plan_to_joints(self, target: Sequence[float], vel_scale: float, acc_scale: float) -> Plan: ...

    @abc.abstractmethod
    async def plan_to_pose(self, pose: Pose, vel_scale: float, acc_scale: float) -> Plan: ...

    @abc.abstractmethod
    async def plan_cartesian(
        self, waypoints: Sequence[Pose], max_step: float, vel_scale: float, acc_scale: float
    ) -> Plan:
        """Straight-line TCP path through ``waypoints`` (base frame), interpolated every ``max_step`` m."""

    # --- execution ---------------------------------------------------------------------
    @abc.abstractmethod
    async def execute(
        self, plan: Plan, on_progress: ProgressCallback, should_abort: AbortCheck
    ) -> ExecutionResult:
        """Run ``plan`` and return when the motion has ended.

        Poll ``should_abort`` while moving and stop promptly when it returns True. If this coroutine
        is cancelled, the motion must be stopped (e.g. cancel the trajectory goal in a shielded
        scope) before the cancellation propagates. The server additionally calls :meth:`stop` on
        cancellation, but must not be the only line of defence.
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop ALL motion - the arm trajectory and any gripper action - as quickly as the controllers
        allow. Must never raise for 'nothing to stop'. Used by ``stop_motion``, ``estop``, the force
        monitor and on cancellation."""

    # --- gripper -----------------------------------------------------------------------
    @abc.abstractmethod
    async def gripper_move(self, width: float, speed: float) -> GripperState: ...

    @abc.abstractmethod
    async def gripper_grasp(
        self, width: float, force: float, speed: float, epsilon_inner: float, epsilon_outer: float
    ) -> GripperState: ...

    @abc.abstractmethod
    async def gripper_home(self) -> GripperState: ...

    @abc.abstractmethod
    async def get_gripper_state(self) -> GripperState | None: ...

    # --- perception --------------------------------------------------------------------
    @abc.abstractmethod
    async def camera_snapshot(self, topic: str, max_width: int) -> CameraFrame: ...

    # --- robot-specific ----------------------------------------------------------------
    @abc.abstractmethod
    async def set_collision_thresholds(self, force_n: float, torque_nm: float) -> None: ...

    @abc.abstractmethod
    async def error_recovery(self) -> None: ...

    @abc.abstractmethod
    async def list_graph(self) -> GraphInfo: ...
