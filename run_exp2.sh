#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/exp2_scene_groups}"
RENDER_ITERATION="${RENDER_ITERATION:-30000}"
PROFILE="${PROFILE:-progressive_full}"
GROUP_JSON="${GROUP_JSON:-${OUTPUT_ROOT}/scene_group_summary.json}"

COMMON_TRAIN_ARGS=(
  --eval
)

COMMON_RENDER_ARGS=(
  --skip_train
)

COMMON_METRICS_ARGS=()

SCENES=()
if [ "$#" -gt 0 ]; then
  SCENES=("$@")
fi

if [ "${#SCENES[@]}" -eq 0 ]; then
  cat <<'EOF'
Usage:
  ./run_exp2.sh /path/to/FUNA/SceneA/with/blue/1 [/path/to/FUNA/SceneB/with/green/2 ...]

This script runs the stratified scene statistics experiment with the full model
and then groups the results by water color, degradation level, depth/background/texture bins.
EOF
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

infer_air_path() {
  local scene="$1"
  local with_dir
  with_dir="$(dirname "$(dirname "$scene")")"
  if [ "$(basename "$with_dir")" = "with" ]; then
    local air_dir="${with_dir}/air"
    if [ -d "$air_dir" ]; then
      printf '%s\n' "$air_dir"
      return 0
    fi
  fi
  printf '%s\n' ""
}

scene_tag() {
  local scene="$1"
  local level color with_dir scene_dir scene_name
  level="$(basename "$scene")"
  color="$(basename "$(dirname "$scene")")"
  with_dir="$(dirname "$(dirname "$scene")")"
  if [ "$(basename "$with_dir")" = "with" ]; then
    scene_dir="$(dirname "$with_dir")"
    scene_name="$(basename "$scene_dir")"
    printf '%s_%s_%s\n' "$scene_name" "$color" "$level"
  else
    printf '%s\n' "$(basename "$scene")"
  fi
}

mkdir -p "$OUTPUT_ROOT"

OUTPUTS=()
for scene in "${SCENES[@]}"; do
  if [ ! -d "$scene" ]; then
    echo "Scene path does not exist: $scene" >&2
    exit 1
  fi

  air_path="$(infer_air_path "$scene")"
  tag="$(scene_tag "$scene")"
  out_dir="${OUTPUT_ROOT}/${tag}"
  OUTPUTS+=("$out_dir")

  train_cmd=(
    "$PYTHON_BIN" train.py
    -s "$scene"
    --model_path "$out_dir"
    --experiment_profile "$PROFILE"
    "${COMMON_TRAIN_ARGS[@]}"
  )
  if [ -n "$air_path" ]; then
    train_cmd+=(--air_source_path "$air_path")
  fi

  echo "[exp2][train] scene=$scene profile=$PROFILE out=$out_dir"
  "${train_cmd[@]}"

  echo "[exp2][render] out=$out_dir"
  "$PYTHON_BIN" render.py \
    --model_path "$out_dir" \
    --iteration "$RENDER_ITERATION" \
    "${COMMON_RENDER_ARGS[@]}"
done

echo "[exp2][metrics] evaluating ${#OUTPUTS[@]} runs"
"$PYTHON_BIN" metrics3.py \
  -m "${OUTPUTS[@]}" \
  "${COMMON_METRICS_ARGS[@]}"

echo "[exp2][group] writing $GROUP_JSON"
"$PYTHON_BIN" scripts/scene_group_analysis.py \
  -m "${OUTPUTS[@]}" \
  --output_json "$GROUP_JSON"

echo "[exp2] done"
