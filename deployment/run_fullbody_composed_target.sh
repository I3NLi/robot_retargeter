#!/usr/bin/env bash
set -eo pipefail

APP_DIR="${KENGO_FULLBODY_COMPOSED_TARGET_ROOT:-/opt/kengo-fullbody-composed-target/current}"
source /opt/ros/humble/setup.bash
source /home/admin/galaxea/install/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-89}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/home/admin/instinct_deepdive_ws/no_shm.xml}"

binary="${APP_DIR}/bin/kengo_fullbody_composed_target_node"
[[ -x "${binary}" ]] || {
  echo "missing composed-target binary: ${binary}" >&2
  exit 66
}
exec "${binary}"
