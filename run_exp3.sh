#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/exp3_fakeair}"
RENDER_ITERATION="${RENDER_ITERATION:-30000}"
ANALYSIS_JSON="${ANALYSIS_JSON:-${OUTPUT_ROOT}/fakeair_summary.json}"

PROFILES=(
  fakeair_none
  fakeair_pseudo
  fakeair_oracle
)

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
  ./run_exp3.sh /path/to/FUNA/SceneA/with/blue/1 [/path/to/FUNA/SceneB/with/green/2 ...]

This script runs the pseudo-clean supervision comparison:
  no pseudo clean / pseudo clean / oracle clean
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

  for profile in "${PROFILES[@]}"; do
    out_dir="${OUTPUT_ROOT}/${tag}_${profile}"
    OUTPUTS+=("$out_dir")

    train_cmd=(
      "$PYTHON_BIN" train.py
      -s "$scene"
      --model_path "$out_dir"
      --experiment_profile "$profile"
      "${COMMON_TRAIN_ARGS[@]}"
    )
    if [ -n "$air_path" ]; then
      train_cmd+=(--air_source_path "$air_path")
    fi

    echo "[exp3][train] scene=$scene profile=$profile out=$out_dir"
    "${train_cmd[@]}"

    echo "[exp3][render] out=$out_dir"
    "$PYTHON_BIN" render.py \
      --model_path "$out_dir" \
      --iteration "$RENDER_ITERATION" \
      "${COMMON_RENDER_ARGS[@]}"
  done
done

echo "[exp3][metrics] evaluating ${#OUTPUTS[@]} runs"
"$PYTHON_BIN" metrics3.py \
  -m "${OUTPUTS[@]}" \
  "${COMMON_METRICS_ARGS[@]}"

echo "[exp3][summary] writing $ANALYSIS_JSON"
"$PYTHON_BIN" scripts/analyze_experiments.py \
  -m "${OUTPUTS[@]}" \
  --analysis fakeair \
  --output_json "$ANALYSIS_JSON"

echo "[exp3] done"
