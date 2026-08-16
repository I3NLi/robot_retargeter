#!/usr/bin/env bash
set -Eeuo pipefail

readonly RETARGET_ROOT="/opt/kengo-fullbody-retarget"
readonly COMPOSED_ROOT="/opt/kengo-fullbody-composed-target"
readonly RETARGET_UNIT="kengo-fullbody-retarget.service"
readonly COMPOSED_UNIT="kengo-fullbody-composed-target.service"
readonly COMMAND_UNIT="kengo-fullbody-command-bridge.service"
readonly LOCK_FILE="/run/lock/kengo-fullbody-retarget-install.lock"
readonly TARGET_USER="admin"
readonly TARGET_GROUP="admin"

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
MUJOCO_ROOT="${KENGO_MUJOCO_ROOT:-}"
release_id="release-$(date -u +%Y%m%dT%H%M%SZ)"
candidate="${RETARGET_ROOT}/releases/${release_id}"
retarget_link="${RETARGET_ROOT}/current"
composed_link="${COMPOSED_ROOT}/current"
old_retarget=""
old_composed=""
retarget_was_active=0
composed_was_active=0
switched=0
committed=0
unit_backup="${candidate}/unit-backup"

log() { printf '[kengo-fullbody] %s\n' "$*"; }
die() { printf '[kengo-fullbody] ERROR: %s\n' "$*" >&2; exit 1; }

(( EUID == 0 )) || die "run with sudo"
[[ "$(uname -s)" == Linux ]] || die "Linux is required"
id "${TARGET_USER}" >/dev/null 2>&1 || die "missing user ${TARGET_USER}"

exec 9>"${LOCK_FILE}"
flock -n 9 || die "another full-body installation is running"

required=(
  cpp/CMakeLists.txt cpp/package.xml
  config/robot/kengo.yaml
  asset/robot/kengo_description/mjcf/kengo.xml
  deployment/build_realtime_fullbody_cpp.sh
  deployment/run_realtime_fullbody.sh
  deployment/run_fullbody_composed_target.sh
  deployment/kengo-fullbody-retarget.service
  deployment/kengo-fullbody-composed-target.service
)
for path in "${required[@]}"; do
  [[ -f "${SOURCE_ROOT}/${path}" && ! -L "${SOURCE_ROOT}/${path}" ]] || die "missing regular payload file: ${path}"
done
mesh_count="$(find "${SOURCE_ROOT}/asset/robot/kengo_description/meshes" -maxdepth 1 -type f -name '*.STL' | wc -l)"
[[ "${mesh_count}" == 27 ]] || die "expected exactly 27 Kengo STL meshes, found ${mesh_count}"

for command_name in cmake curl find flock install python3 systemctl tar; do
  command -v "${command_name}" >/dev/null 2>&1 || die "missing command: ${command_name}"
done

safe_robot_state() {
  local mode walk
  mode="$(curl -fsS --max-time 5 http://127.0.0.1:8000/motion/get_mode)" || return 1
  python3 -c 'import json,sys; p=json.load(sys.stdin); raise SystemExit(0 if p.get("success") is True and p.get("current_mode")==1 else 1)' <<<"${mode}" || return 1
  walk="$(curl -fsS --max-time 5 http://127.0.0.1:8088/api/instinct/walk)" || return 1
  python3 -c 'import json,sys; p=json.load(sys.stdin); raise SystemExit(0 if p.get("success") is True and p.get("running") is False and p.get("pid") is None else 1)' <<<"${walk}" || return 1
  ! pgrep -f '[R]8_beyondmimic_multi_agent.py|[l]aunch_kengo_walk_nodryrun' >/dev/null
}

discover_mujoco_root() {
  local python_bin candidate_header
  for python_bin in \
    /opt/kengo-fullbody-retarget/venv/bin/python3 \
    /home/admin/robot_retargeter/.venv/bin/python3 \
    /usr/bin/python3; do
    [[ -x "${python_bin}" ]] || continue
    MUJOCO_ROOT="$(${python_bin} -c 'import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)' 2>/dev/null || true)"
    [[ -f "${MUJOCO_ROOT}/include/mujoco/mujoco.h" && -f "${MUJOCO_ROOT}/libmujoco.so.3.3.4" ]] && return 0
  done
  candidate_header="$(find /opt /home/admin -xdev -type f -path '*/mujoco/include/mujoco/mujoco.h' -print -quit 2>/dev/null || true)"
  [[ -n "${candidate_header}" ]] || return 1
  MUJOCO_ROOT="${candidate_header%/include/mujoco/mujoco.h}"
  [[ -f "${MUJOCO_ROOT}/libmujoco.so.3.3.4" ]]
}

if [[ -n "${MUJOCO_ROOT}" ]]; then
  MUJOCO_ROOT="$(cd -- "${MUJOCO_ROOT}" && pwd -P)"
  [[ -f "${MUJOCO_ROOT}/include/mujoco/mujoco.h" && -f "${MUJOCO_ROOT}/libmujoco.so.3.3.4" ]] || die "invalid KENGO_MUJOCO_ROOT"
else
  discover_mujoco_root || die "MuJoCo 3.3.4 development package was not found; set KENGO_MUJOCO_ROOT"
fi

safe_robot_state || die "Walk must be stopped and HDAS must be in Mode 1"
install -d -m 0755 "${RETARGET_ROOT}/releases" "${COMPOSED_ROOT}"
[[ ! -e "${candidate}" && ! -L "${candidate}" ]] || die "candidate already exists: ${candidate}"
install -d -o "${TARGET_USER}" -g "${TARGET_GROUP}" -m 0755 "${candidate}"

log "staging deterministic candidate ${release_id}"
for directory in cpp config asset deployment; do
  cp -a -- "${SOURCE_ROOT}/${directory}" "${candidate}/${directory}"
done
chown -R "${TARGET_USER}:${TARGET_GROUP}" "${candidate}"
chmod 0755 "${candidate}/deployment/"*.sh

log "building and self-testing native telemetry nodes"
KENGO_CPP_BUILD_JOBS="${KENGO_CPP_BUILD_JOBS:-2}" \
  "${candidate}/deployment/build_realtime_fullbody_cpp.sh" \
  "${candidate}" "${candidate}" "${MUJOCO_ROOT}" "${candidate}/build"
[[ -x "${candidate}/bin/kengo_fullbody_retarget_node" ]] || die "retarget binary is missing"
[[ -x "${candidate}/bin/kengo_fullbody_composed_target_node" ]] || die "composed-target binary is missing"
[[ ! -e "${candidate}/bin/kengo_fullbody_command_bridge_node" ]] || die "archived command publisher must not be built"
safe_robot_state || die "robot safety state changed during candidate build"

[[ ! -e "${retarget_link}" || -L "${retarget_link}" ]] || die "retarget current path is not a symlink"
[[ ! -e "${composed_link}" || -L "${composed_link}" ]] || die "composed current path is not a symlink"
old_retarget="$(readlink "${retarget_link}" 2>/dev/null || true)"
old_composed="$(readlink "${composed_link}" 2>/dev/null || true)"
systemctl is-active --quiet "${RETARGET_UNIT}" && retarget_was_active=1 || true
systemctl is-active --quiet "${COMPOSED_UNIT}" && composed_was_active=1 || true
install -d -m 0700 "${unit_backup}"
for unit in "${RETARGET_UNIT}" "${COMPOSED_UNIT}"; do
  [[ -f "/etc/systemd/system/${unit}" ]] && cp -a -- "/etc/systemd/system/${unit}" "${unit_backup}/${unit}"
done

atomic_link() {
  local target="$1" link="$2" temporary
  temporary="${link}.next.$$"
  ln -s -- "${target}" "${temporary}"
  mv -Tf -- "${temporary}" "${link}"
}

restore_link() {
  local previous="$1" link="$2"
  if [[ -n "${previous}" ]]; then atomic_link "${previous}" "${link}"; else rm -f -- "${link}"; fi
}

rollback() {
  [[ "${switched}" == 1 && "${committed}" == 0 ]] || return 0
  log "candidate failed; restoring previous release links"
  systemctl stop "${COMPOSED_UNIT}" "${RETARGET_UNIT}" 2>/dev/null || true
  restore_link "${old_retarget}" "${retarget_link}"
  restore_link "${old_composed}" "${composed_link}"
  for unit in "${RETARGET_UNIT}" "${COMPOSED_UNIT}"; do
    if [[ -f "${unit_backup}/${unit}" ]]; then
      cp -a -- "${unit_backup}/${unit}" "/etc/systemd/system/${unit}"
    else
      rm -f -- "/etc/systemd/system/${unit}"
    fi
  done
  systemctl daemon-reload
  (( retarget_was_active == 1 )) && systemctl start "${RETARGET_UNIT}" || true
  (( composed_was_active == 1 )) && systemctl start "${COMPOSED_UNIT}" || true
}
trap rollback EXIT
trap 'exit 130' INT TERM HUP

log "archiving the former direct command publisher"
systemctl disable --now "${COMMAND_UNIT}" >/dev/null 2>&1 || true
safe_robot_state || die "robot safety state changed immediately before activation"

systemctl stop "${COMPOSED_UNIT}" "${RETARGET_UNIT}" 2>/dev/null || true
install -m 0644 "${candidate}/deployment/${RETARGET_UNIT}" "/etc/systemd/system/${RETARGET_UNIT}"
install -m 0644 "${candidate}/deployment/${COMPOSED_UNIT}" "/etc/systemd/system/${COMPOSED_UNIT}"
atomic_link "${candidate}" "${retarget_link}"
atomic_link "${candidate}" "${composed_link}"
switched=1
systemctl daemon-reload
systemctl enable "${RETARGET_UNIT}" "${COMPOSED_UNIT}" >/dev/null
systemctl restart "${RETARGET_UNIT}" "${COMPOSED_UNIT}"

healthy_samples=0
for _ in {1..30}; do
  if systemctl is-active --quiet "${RETARGET_UNIT}" && systemctl is-active --quiet "${COMPOSED_UNIT}"; then
    ((healthy_samples += 1))
    if (( healthy_samples >= 4 )); then
      committed=1
      log "installation complete: ${candidate}"
      exit 0
    fi
  else
    healthy_samples=0
  fi
  sleep 1
done
systemctl status "${RETARGET_UNIT}" "${COMPOSED_UNIT}" --no-pager || true
journalctl -u "${RETARGET_UNIT}" -u "${COMPOSED_UNIT}" -n 80 --no-pager || true
die "candidate services did not become active"
