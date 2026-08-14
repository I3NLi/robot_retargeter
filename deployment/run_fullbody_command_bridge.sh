#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${KENGO_FULLBODY_COMMAND_ARMED:-0}" != "1" ]]; then
  echo "KENGO_FULLBODY_COMMAND_ARMED=1 is required to claim /hybrid_body_controller/commands" >&2
  exit 77
fi

RELEASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
set +u
source /opt/ros/humble/setup.bash
source /home/admin/galaxea/install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-89}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/home/admin/instinct_deepdive_ws/no_shm.xml}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/run/kengo-fullbody-command-bridge}"
export MALLOC_ARENA_MAX=2

exec "${RELEASE_DIR}/bin/kengo_fullbody_command_bridge_node"
