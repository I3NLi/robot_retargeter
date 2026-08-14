#!/usr/bin/env bash
set -Eeuo pipefail

RELEASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
set +u
source /opt/ros/humble/setup.bash
source /home/admin/galaxea/install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-89}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/home/admin/instinct_deepdive_ws/no_shm.xml}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/run/kengo-fullbody-retarget}"
export LD_LIBRARY_PATH="${RELEASE_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MALLOC_ARENA_MAX=2

exec "${RELEASE_DIR}/bin/kengo_fullbody_retarget_node" \
  --model "${RELEASE_DIR}/asset/robot/kengo_description/mjcf/kengo.xml" \
  --config "${RELEASE_DIR}/config/robot/kengo.yaml"
