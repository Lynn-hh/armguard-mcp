#!/usr/bin/env bash
# EXPERIMENTAL / UNTESTED: source ROS 2 (and an optional overlay), then run armguard-mcp.
# Nothing may be printed to stdout here: on the stdio transport stdout carries MCP messages.
set -eo pipefail
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" >&2
if [[ -n "${ARMGUARD_EXTRA_SETUP:-}" && -f "${ARMGUARD_EXTRA_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${ARMGUARD_EXTRA_SETUP}" >&2
fi
if [[ "${1:-}" == "--" ]]; then
  shift
  exec "$@"          # run an arbitrary command, e.g. `-- ros2 launch ...`
fi
exec /opt/venv/bin/armguard-mcp "$@"
