#!/usr/bin/env bash
set -uo pipefail

# Batch the already-retargeted LaFAN1 G1 CSVs through the validated
# G1 -> Kengo keypoint/IK pipeline, then optionally export 50 Hz WBT NPZs.
#
# Completion is clip-atomic at the NPZ level: an existing clip is skipped only
# after validate_kengo_motion.py passes.  This makes interrupted runs safe to
# resume without trusting partial CSV or NPZ files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETARGET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WBT_ROOT="${WBT_ROOT:-/home/hiyio/by_robot/kengo_wbt}"
INPUT_DIR="${INPUT_DIR:-${RETARGET_ROOT}/dataset/lafan1_g1}"
NPZ_DIR="${NPZ_DIR:-${WBT_ROOT}/data/lafan1_kengo}"
LOG_DIR="${LOG_DIR:-${WBT_ROOT}/logs/lafan1_kengo_retarget}"
GMR_PYTHON="${GMR_PYTHON:-/home/hiyio/anaconda3/envs/gmr/bin/python}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python}"
SOURCE_FPS="${SOURCE_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
KENGO_URDF_PATH="${KENGO_URDF_PATH:-/home/hiyio/by_robot/kengo资料/URDF文件/galaxea_robot_assets-main/galaxea_robot_assets/kengo_with_fist/urdf/kengo_with_fist.urdf}"

STAGE="all"
FORCE=false
DRY_RUN=false
LIMIT=0

usage() {
  cat <<'EOF'
Usage: retarget_lafan1_to_kengo_batch.sh [options]

Options:
  --stage all|retarget|npz  Run the complete chain or one stage (default: all).
  --force                   Ignore an already-valid NPZ and rebuild the clip.
  --dry-run                 Print the selected inputs and commands only.
  --limit N                 Process at most N sorted clips (0 means all).
  -h, --help                Show this help.

Environment overrides:
  INPUT_DIR, NPZ_DIR, LOG_DIR, WBT_ROOT, GMR_PYTHON, ISAAC_PYTHON,
  SOURCE_FPS, OUTPUT_FPS, KENGO_URDF_PATH.
EOF
}

while (($#)); do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || { echo "[error] --stage needs a value" >&2; exit 2; }
      STAGE="$2"
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --limit)
      [[ $# -ge 2 ]] || { echo "[error] --limit needs a value" >&2; exit 2; }
      LIMIT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$STAGE" in
  all|retarget|npz) ;;
  *) echo "[error] --stage must be all, retarget, or npz" >&2; exit 2 ;;
esac
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "[error] --limit must be a non-negative integer" >&2; exit 2; }
[[ -d "$INPUT_DIR" ]] || { echo "[error] input directory not found: $INPUT_DIR" >&2; exit 2; }
[[ -d "$WBT_ROOT" ]] || { echo "[error] WBT root not found: $WBT_ROOT" >&2; exit 2; }
[[ -x "$GMR_PYTHON" ]] || { echo "[error] GMR Python is not executable: $GMR_PYTHON" >&2; exit 2; }
[[ -x "$ISAAC_PYTHON" ]] || { echo "[error] Isaac Python is not executable: $ISAAC_PYTHON" >&2; exit 2; }
[[ -f "$KENGO_URDF_PATH" ]] || { echo "[error] Kengo URDF not found: $KENGO_URDF_PATH" >&2; exit 2; }

mkdir -p "$NPZ_DIR" "$LOG_DIR"

mapfile -d '' -t SOURCES < <(
  find "$INPUT_DIR" -maxdepth 1 -type f -name '*.csv' \
    ! -name 'Form_1_stageii_g1.csv' -print0 | sort -z
)
if ((${#SOURCES[@]} == 0)); then
  echo "[error] no LaFAN1 CSV inputs found under: $INPUT_DIR" >&2
  exit 2
fi
if ((LIMIT > 0 && LIMIT < ${#SOURCES[@]})); then
  SOURCES=("${SOURCES[@]:0:LIMIT}")
fi

export OMNI_KIT_ACCEPT_EULA=YES
export KENGO_URDF_PATH
export PYTHONUNBUFFERED=1

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

validate_npz() {
  local npz_path="$1"
  local log_path="$2"
  [[ -s "$npz_path" ]] || return 1
  "$ISAAC_PYTHON" "$WBT_ROOT/scripts/validate_kengo_motion.py" "$npz_path" >>"$log_path" 2>&1
}

failures=()
passed=0
skipped=0
total=${#SOURCES[@]}

echo "[batch] stage=$STAGE clips=$total input=$INPUT_DIR"
echo "[batch] csv_dir=$RETARGET_ROOT/output_data/robot_motion"
echo "[batch] npz_dir=$NPZ_DIR logs=$LOG_DIR"

for index in "${!SOURCES[@]}"; do
  src="${SOURCES[$index]}"
  filename="${src##*/}"
  stem="${filename%.csv}"
  keypoints="$RETARGET_ROOT/output_data/keypoints/kengo/${stem}_from_g1_keypoints.pkl"
  csv="$RETARGET_ROOT/output_data/robot_motion/${stem}_from_g1_kengo.csv"
  npz="$NPZ_DIR/${stem}_kengo_fixed_50hz.npz"
  log="$LOG_DIR/${stem}.log"
  ordinal=$((index + 1))

  printf '[%d/%d] %s\n' "$ordinal" "$total" "$stem"

  if [[ "$FORCE" == false && "$DRY_RUN" == false ]] && validate_npz "$npz" "$log"; then
    echo "  [skip PASS] $npz"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$STAGE" == all || "$STAGE" == retarget ]]; then
    replay_cmd=(
      "$GMR_PYTHON" scripts/robot_replay.py
      --no-viewer
      --motion-file "$src"
      --source-robot-config config/robot/g1.yaml
      --target-robot-config config/robot/kengo.yaml
      --fps "$SOURCE_FPS"
    )
    retarget_cmd=(
      "$GMR_PYTHON" scripts/robot_retarget.py
      --config config/robot/kengo.yaml
      --keypoints-name "${stem}_from_g1"
      --no-render-debug
    )

    if [[ "$DRY_RUN" == true ]]; then
      print_command "${replay_cmd[@]}"
      print_command "${retarget_cmd[@]}"
    else
      {
        printf '\n[%s] replay %s\n' "$(date --iso-8601=seconds)" "$stem"
        printf 'source=%s\nkeypoints=%s\ncsv=%s\n' "$src" "$keypoints" "$csv"
      } >>"$log"
      if ! (cd "$RETARGET_ROOT" && "${replay_cmd[@]}") >>"$log" 2>&1; then
        echo "  [FAIL] robot_replay; see $log" >&2
        failures+=("$stem:robot_replay")
        continue
      fi
      if ! (cd "$RETARGET_ROOT" && "${retarget_cmd[@]}") >>"$log" 2>&1; then
        echo "  [FAIL] robot_retarget; see $log" >&2
        failures+=("$stem:robot_retarget")
        continue
      fi
      if [[ ! -s "$keypoints" || ! -s "$csv" ]]; then
        echo "  [FAIL] retarget outputs missing; see $log" >&2
        failures+=("$stem:retarget_outputs")
        continue
      fi
      echo "  [retarget PASS] $csv"
    fi
  elif [[ ! -s "$csv" ]]; then
    echo "  [FAIL] NPZ stage requires CSV: $csv" >&2
    failures+=("$stem:missing_csv")
    continue
  fi

  if [[ "$STAGE" == all || "$STAGE" == npz ]]; then
    convert_cmd=(
      "$ISAAC_PYTHON" scripts/csv_to_npz.py
      --robot kengo
      --input_file "$csv"
      --input_joint_order hardware
      --input_fps "$SOURCE_FPS"
      --output_file "$npz"
      --output_fps "$OUTPUT_FPS"
      --headless
    )
    validate_cmd=("$ISAAC_PYTHON" scripts/validate_kengo_motion.py "$npz")

    if [[ "$DRY_RUN" == true ]]; then
      print_command "${convert_cmd[@]}"
      print_command "${validate_cmd[@]}"
    else
      printf '[%s] convert %s\n' "$(date --iso-8601=seconds)" "$stem" >>"$log"
      if ! (cd "$WBT_ROOT" && "${convert_cmd[@]}") >>"$log" 2>&1; then
        echo "  [FAIL] csv_to_npz; see $log" >&2
        failures+=("$stem:csv_to_npz")
        continue
      fi
      if ! validate_npz "$npz" "$log"; then
        echo "  [FAIL] validate_kengo_motion; see $log" >&2
        failures+=("$stem:validate")
        continue
      fi
      echo "  [PASS] $npz"
      passed=$((passed + 1))
    fi
  elif [[ "$DRY_RUN" == false ]]; then
    passed=$((passed + 1))
  fi
done

echo "[summary] selected=$total passed=$passed skipped=$skipped failures=${#failures[@]}"
if ((${#failures[@]})); then
  printf '[failure] %s\n' "${failures[@]}"
  exit 1
fi
