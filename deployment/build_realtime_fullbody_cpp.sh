#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 SOURCE_ROOT RELEASE_DIR MUJOCO_ROOT [BUILD_DIR]" >&2
  exit 64
fi

SOURCE_ROOT="$(cd -- "$1" && pwd -P)"
RELEASE_DIR="$2"
MUJOCO_ROOT="$(cd -- "$3" && pwd -P)"
BUILD_DIR="${4:-${SOURCE_ROOT}/build/fullbody-cpp}"

[[ -f "${SOURCE_ROOT}/cpp/CMakeLists.txt" ]]
[[ -f "${MUJOCO_ROOT}/include/mujoco/mujoco.h" ]]
[[ -f "${MUJOCO_ROOT}/libmujoco.so.3.3.4" ]]
mkdir -p -- "$BUILD_DIR" "$RELEASE_DIR/bin" "$RELEASE_DIR/lib"

set +u
source /opt/ros/humble/setup.bash
source /home/admin/galaxea/install/setup.bash
set -u

cmake -S "${SOURCE_ROOT}/cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LEGACY_COMMAND_BRIDGE=OFF \
  -DMUJOCO_ROOT="$MUJOCO_ROOT"
cmake --build "$BUILD_DIR" --parallel "${KENGO_CPP_BUILD_JOBS:-2}"
ctest --test-dir "$BUILD_DIR" --output-on-failure

cmake --install "$BUILD_DIR" --prefix "$RELEASE_DIR"
install -m 0644 -- "$MUJOCO_ROOT/libmujoco.so.3.3.4" \
  "$RELEASE_DIR/lib/libmujoco.so.3.3.4"

LD_LIBRARY_PATH="${RELEASE_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "$RELEASE_DIR/bin/kengo_fullbody_retarget_node" \
  --model "$RELEASE_DIR/asset/robot/kengo_description/mjcf/kengo.xml" \
  --config "$RELEASE_DIR/config/robot/kengo.yaml" \
  --self-test --benchmark-frames 30
