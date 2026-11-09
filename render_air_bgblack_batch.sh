#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Default paths
DATA_ROOT="${DATA_ROOT:-/mnt/d/home/ycy}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-outputs/funa_exp13_6gpu}"

# GPUs
GPUS=(4 5 6 7 8 9)

# Render arguments
RENDER_ARGS=(
  --skip_train
  --air_bg_black 1
  --air_bg_use_hit_map 1
)

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

# Function to parse scene path from model path
parse_scene_path() {
  local model_path="$1"
  local tag
  tag="$(basename "$model_path" | sed 's/_progressive_stage.*\|_fakeair_.*//')"
  # tag like Bay_submerged_ships_2_blue_4
  local scene_name="${tag%_blue_4}"
  local color="blue"
  local level="4"
  local scene_path="${DATA_ROOT}/${scene_name}/with/${color}/${level}"
  if [ -d "$scene_path" ]; then
    printf '%s\n' "$scene_path"
  else
    echo "[warn] scene path not found: $scene_path" >&2
    printf '%s\n' ""
  fi
}

# Collect all model paths
MODEL_PATHS=()
while IFS= read -r path; do
  MODEL_PATHS+=("$path")
done < <(find "${OUTPUT_ROOT_BASE}" -mindepth 3 -maxdepth 3 -type d \( -name "*progressive_stage*" -o -name "*fakeair_*" \) | sort)

if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
  echo "[error] no model paths found" >&2
  exit 1
fi

echo "[info] found ${#MODEL_PATHS[@]} model paths to render and evaluate"

# Launch workers
idx=0
for model_path in "${MODEL_PATHS[@]}"; do
  gpu_id="${GPUS[$((idx % ${#GPUS[@]}))]}"
  idx=$((idx + 1))

  scene_path="$(parse_scene_path "$model_path")"
  if [ -z "$scene_path" ]; then
    echo "[skip] invalid scene path for $model_path"
    continue
  fi

  log_path="${model_path}/render_eval.log"
  echo "[worker] gpu=${gpu_id} model=${model_path} scene=${scene_path}"
  cd "$REPO_DIR"
  env CUDA_VISIBLE_DEVICES="$gpu_id" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" render.py -s "$scene_path" -m "$model_path" "${RENDER_ARGS[@]}"
  cd "$model_path"
  env CUDA_VISIBLE_DEVICES="$gpu_id" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" "$REPO_DIR/metrics.py" -m .
  echo "[done] $model_path"
done

echo "[done] all renders and evaluations completed"