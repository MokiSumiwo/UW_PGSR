#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Default FUNA root inferred from your existing scripts.
DATA_ROOT="${DATA_ROOT:-/mnt/d/home/ycy}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-outputs/funa_exp13_6gpu}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT_BASE}/logs}"

# One base FUNA scene per GPU.
SCENES=(
  Bay_submerged_ships_2
  Egypt_Ruins_Environment_2
  Eternal_temple_3
  oceanEnvironmentPark_2
  V3_Cave02
  V3_Dive3
)

GPUS=(4 5 6 7 8 9)
COLORS=(blue)
LEVELS=(4)

# Common knobs passed down to child scripts.
RENDER_ITERATION="${RENDER_ITERATION:-30000}"
RUN_EXP1="${RUN_EXP1:-0}"
RUN_EXP3="${RUN_EXP3:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

if [ "${#SCENES[@]}" -ne "${#GPUS[@]}" ]; then
  echo "SCENES and GPUS must have the same length." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT_BASE" "$LOG_ROOT"

collect_scene_variants() {
  local scene_name="$1"
  local -n out_ref="$2"
  local color level scene_path

  out_ref=()
  for color in "${COLORS[@]}"; do
    for level in "${LEVELS[@]}"; do
      scene_path="${DATA_ROOT}/${scene_name}/with/${color}/${level}"
      if [ -d "$scene_path" ]; then
        out_ref+=("$scene_path")
      else
        echo "[warn] missing scene variant, skip: $scene_path" >&2
      fi
    done
  done
}

launch_scene_worker() {
  local scene_name="$1"
  local gpu_id="$2"
  local log_path="${LOG_ROOT}/${scene_name}_exp3.log"
  local -a variants=()

  collect_scene_variants "$scene_name" variants
  if [ "${#variants[@]}" -eq 0 ]; then
    echo "[fatal] no valid variants found for scene ${scene_name}" >&2
    return 1
  fi

  local variants_cmd
  variants_cmd="$(printf '%q ' "${variants[@]}")"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    PYTHON_BIN="$PYTHON_BIN" \
    RENDER_ITERATION="$RENDER_ITERATION" \
    OUTPUT_ROOT_BASE="$OUTPUT_ROOT_BASE" \
    RUN_EXP1="$RUN_EXP1" \
    RUN_EXP3="$RUN_EXP3" \
    DATA_ROOT="$DATA_ROOT" \
    scene_name="$scene_name" \
    gpu_id="$gpu_id" \
    variants_cmd="$variants_cmd" \
    bash -lc '
    set -euo pipefail
    eval "variants=($variants_cmd)"

    echo "[worker] scene=${scene_name} gpu=${gpu_id} variants=${#variants[@]}"

    if [ "$RUN_EXP1" = "1" ]; then
      export OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/${scene_name}/exp1"
      export ANALYSIS_JSON="${OUTPUT_ROOT_BASE}/${scene_name}/exp1_progressive_summary.json"
      echo "[worker][${scene_name}] start exp1 on gpu ${gpu_id}"
      ./run_exp1.sh "${variants[@]}"
    fi

    if [ "$RUN_EXP3" = "1" ]; then
      export OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/${scene_name}/exp3"
      export ANALYSIS_JSON="${OUTPUT_ROOT_BASE}/${scene_name}/exp3_fakeair_summary.json"
      echo "[worker][${scene_name}] start exp3 on gpu ${gpu_id}"
      ./run_exp3.sh "${variants[@]}"
    fi

    echo "[worker] scene=${scene_name} gpu=${gpu_id} finished
  ' >"$log_path" 2>&1 &

  PIDS+=("$!")
}

PIDS=()
for idx in "${!SCENES[@]}"; do
  scene_name="${SCENES[$idx]}"
  gpu_id="${GPUS[$idx]}"
  launch_scene_worker "$scene_name" "$gpu_id"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "[fatal] at least one scene worker failed. Check logs under ${LOG_ROOT}" >&2
  exit 1
fi

aggregate_outputs() {
  local exp_name="$1"
  local analysis_type="$2"
  local summary_json="$3"
  local find_pattern="$4"
  local -a model_paths=()

  while IFS= read -r path; do
    model_paths+=("$path")
  done < <(find "${OUTPUT_ROOT_BASE}" -mindepth 3 -maxdepth 3 -type d -name "${find_pattern}" | sort)

  if [ "${#model_paths[@]}" -eq 0 ]; then
    echo "[warn] no model paths found for ${exp_name}, skip aggregate summary"
    return 0
  fi

  echo "[aggregate] ${exp_name}: ${#model_paths[@]} runs -> ${summary_json}"
  "$PYTHON_BIN" scripts/analyze_experiments.py     -m "${model_paths[@]}"     --analysis "${analysis_type}"     --output_json "${summary_json}"
}

if [ "$RUN_EXP1" = "1" ]; then
  aggregate_outputs "exp1" "progressive" "${OUTPUT_ROOT_BASE}/exp1_all_progressive_summary.json" "*progressive_stage*"
fi

if [ "$RUN_EXP3" = "1" ]; then
  aggregate_outputs "exp3" "fakeair" "${OUTPUT_ROOT_BASE}/exp3_all_fakeair_summary.json" "*fakeair_*"
fi

echo "[done] all FUNA exp3 jobs finished"
echo "[done] logs: ${LOG_ROOT}"
echo "[done] outputs: ${OUTPUT_ROOT_BASE}"