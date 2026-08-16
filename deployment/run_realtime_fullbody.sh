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

UPPER_FOLLOW_CONFIG="${KENGO_UPPER_FOLLOW_CONFIG:-/var/lib/kengo-robot-gui/upper_follow_speed_rad_s}"
UPPER_FOLLOW_SPEED="0.60"
if [[ -r "${UPPER_FOLLOW_CONFIG}" ]]; then
  IFS= read -r candidate < "${UPPER_FOLLOW_CONFIG}" || true
  if [[ "${candidate:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
      awk -v value="${candidate}" 'BEGIN { exit !(value >= 0.10 && value <= 2.00) }'; then
    UPPER_FOLLOW_SPEED="${candidate}"
  else
    printf '[kengo-fullbody-retarget] ignoring invalid upper follow speed: %s\n' \
      "${candidate:-<empty>}" >&2
  fi
fi

exec "${RELEASE_DIR}/bin/kengo_fullbody_retarget_node" \
  --model "${RELEASE_DIR}/asset/robot/kengo_description/mjcf/kengo.xml" \
  --config "${RELEASE_DIR}/config/robot/kengo.yaml" \
  --ros-args -p "max_upper_follow_velocity_rad_s:=${UPPER_FOLLOW_SPEED}"
