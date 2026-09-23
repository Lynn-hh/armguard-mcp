# syntax=docker/dockerfile:1
#
# armguard-mcp on ROS 2 Jazzy.
#
# !!! EXPERIMENTAL / UNTESTED !!!
# This file has not been built or run by the author yet (no Docker daemon was available when it was
# written). Treat it as a starting point and expect to fix things. Nothing here is a substitute for the
# robot's own safety system or a hardware e-stop.
#
# Targets:
#   armguard   (default)  ROS 2 Jazzy + armguard-mcp in a venv that can see the system rclpy.
#   franka-sim            Adds franka_ros2 built from source, for the fake-hardware + MoveIt demo in
#                         compose.demo.yml. Large and slow to build.
#
#   docker build -f docker/Dockerfile -t armguard-mcp .
#   docker run --rm -i armguard-mcp --policy /opt/armguard/examples/policies/fr3.yaml --backend fake

ARG ROS_DISTRO=jazzy

# ----------------------------------------------------------------------------------------------------
FROM ros:${ROS_DISTRO} AS armguard
ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3-venv python3-pip \
      ros-${ROS_DISTRO}-moveit-msgs \
      ros-${ROS_DISTRO}-control-msgs \
      ros-${ROS_DISTRO}-controller-manager-msgs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/armguard
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples

# --system-site-packages so the venv sees rclpy and the message packages from /opt/ros.
RUN python3 -m venv --system-site-packages /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir ".[image]"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 armguard
USER armguard
# Audit-log directory (compose mounts a named volume here; it inherits this ownership).
RUN mkdir -p /home/armguard/audit

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]

# ----------------------------------------------------------------------------------------------------
FROM armguard AS franka-sim
ARG ROS_DISTRO
# franka_ros2 v3.x targets Jazzy. Pin a released tag for reproducibility; "jazzy" is a moving branch.
ARG FRANKA_ROS2_REF=jazzy
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git python3-colcon-common-extensions python3-vcstool python3-rosdep build-essential \
      ros-${ROS_DISTRO}-moveit ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/franka_ws
RUN git clone --depth 1 --branch ${FRANKA_ROS2_REF} https://github.com/frankarobotics/franka_ros2.git src/franka_ros2 \
 && if [ -f src/franka_ros2/dependency.repos ]; then vcs import src < src/franka_ros2/dependency.repos; fi \
 && (rosdep init 2>/dev/null || true) && rosdep update --rosdistro ${ROS_DISTRO} \
 && apt-get update \
 && rosdep install --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} -y \
 && rm -rf /var/lib/apt/lists/* \
 && . /opt/ros/${ROS_DISTRO}/setup.sh \
 && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
ENV ARMGUARD_EXTRA_SETUP=/opt/franka_ws/install/setup.bash
USER armguard
