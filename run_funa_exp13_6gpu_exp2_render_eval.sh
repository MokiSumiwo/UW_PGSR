#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-${REPO_DIR}/outputs/funa_exp13_6gpu}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/home/ycy}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT_BASE}/logs/exp2_render_eval}"
RENDER_ITERATION="${RENDER_ITERATION:--1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"

usage() {
  cat <<'EOF'
Usage:
  ./run_funa_exp13_6gpu_exp2_render_eval.sh
  ./run_funa_exp13_6gpu_exp2_render_eval.sh /path/to/scene/exp2
  ./run_funa_exp13_6gpu_exp2_render_eval.sh /path/to/model_dir [/path/to/model_dir ...]

Default behavior:
  - auto-discover all model dirs under outputs/funa_exp13_6gpu/*/exp2/*
  - render test views with:
      render.py --skip_train --air_bg_black 1 --air_bg_use_hit_map 1
  - evaluate each model dir with metrics3.py using:
      underwater GT: /mnt/d/home/ycy/<scene>/with/<color>/<level>
      air GT:        /mnt/d/home/ycy/<scene>/with/air

Optional env vars:
  PYTHON_BIN=python3
  OUTPUT_ROOT_BASE=/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu
  DATA_ROOT=/mnt/d/home/ycy
  LOG_ROOT=/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu/logs/exp2_render_eval
  RENDER_ITERATION=-1
  RUN_RENDER=1
  RUN_METRICS=1
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"

expand_target() {
  local target="$1"

  if [ ! -d "$target" ]; then
    echo "[warn] target does not exist: $target" >&2
    return 0
  fi

  if [ -f "$target/cfg_args" ]; then
    printf '%s\n' "$target"
    return 0
  fi

  find "$target" -type d -path '*/exp2/*' ! -path '*/exp2/*/*' | sort
}

parse_model_name() {
  local model_name="$1"
  MODEL_LEVEL="${model_name##*_}"
  local rest="${model_name%_*}"
  MODEL_COLOR="${rest##*_}"
  MODEL_SCENE="${rest%_*}"
}

build_log_tag() {
  local model_dir="$1"
  local rel="${model_dir#${REPO_DIR}/}"
  printf '%s\n' "$rel" | tr '/ ' '__'
}

run_render() {
  local model_dir="$1"
  local log_path="$2"
  local -a cmd=(
    "$PYTHON_BIN" render.py
    --model_path "$model_dir"
    --iteration "$RENDER_ITERATION"
    --skip_train
    --air_bg_black 1
    --air_bg_use_hit_map 1
  )

  {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] render start"
    printf 'Command: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] render done"
  } >>"$log_path" 2>&1
}

run_metrics() {
  local model_dir="$1"
  local source_path="$2"
  local air_source_path="$3"
  local log_path="$4"
  local -a cmd=(
    "$PYTHON_BIN" metrics3.py
    -m "$model_dir"
    -s "$source_path"
  )

  if [ -d "$air_source_path" ]; then
    cmd+=(-sa "$air_source_path")
  else
    echo "[warn] air GT not found, metrics will rely on auto inference or skip AIR full-reference: $air_source_path" >>"$log_path"
  fi

  {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] metrics start"
    printf 'Command: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] metrics done"
  } >>"$log_path" 2>&1
}

if [ "$#" -gt 0 ]; then
  mapfile -t MODEL_DIRS < <(
    for target in "$@"; do
      expand_target "$target"
    done | sort -u
  )
else
  mapfile -t MODEL_DIRS < <(
    find "$OUTPUT_ROOT_BASE" -type d -path '*/exp2/*' ! -path '*/exp2/*/*' | sort
  )
fi

if [ "${#MODEL_DIRS[@]}" -eq 0 ]; then
  echo "No exp2 model directories found." >&2
  exit 1
fi

echo "[info] repo: $REPO_DIR"
echo "[info] output root: $OUTPUT_ROOT_BASE"
echo "[info] data root: $DATA_ROOT"
echo "[info] render iteration: $RENDER_ITERATION"
echo "[info] run render: $RUN_RENDER"
echo "[info] run metrics: $RUN_METRICS"
echo "[info] discovered exp2 models: ${#MODEL_DIRS[@]}"

FAILURES=()
DONE=0
TOTAL="${#MODEL_DIRS[@]}"

for model_dir in "${MODEL_DIRS[@]}"; do
  DONE=$((DONE + 1))
  model_name="$(basename "$model_dir")"
  parse_model_name "$model_name"

  source_path="${DATA_ROOT}/${MODEL_SCENE}/with/${MODEL_COLOR}/${MODEL_LEVEL}"
  air_source_path="${DATA_ROOT}/${MODEL_SCENE}/with/air"
  log_tag="$(build_log_tag "$model_dir")"
  log_path="${LOG_ROOT}/${log_tag}.log"

  echo "[${DONE}/${TOTAL}] model=${model_name}"
  echo "[${DONE}/${TOTAL}] source_path=${source_path}"
  echo "[${DONE}/${TOTAL}] air_source_path=${air_source_path}"
  echo "[${DONE}/${TOTAL}] log=${log_path}"

  if [[ ! "$MODEL_LEVEL" =~ ^[0-9]+$ ]]; then
    echo "[warn] skip invalid model name (level parse failed): $model_dir" >&2
    FAILURES+=("${model_dir} [parse-level]")
    continue
  fi

  if [[ "$MODEL_COLOR" != "blue" && "$MODEL_COLOR" != "green" ]]; then
    echo "[warn] skip invalid model name (color parse failed): $model_dir" >&2
    FAILURES+=("${model_dir} [parse-color]")
    continue
  fi

  if [ ! -d "$source_path" ]; then
    echo "[warn] underwater GT not found: $source_path" >&2
    FAILURES+=("${model_dir} [missing-underwater-gt]")
    continue
  fi

  if [ "$RUN_RENDER" = "1" ]; then
    echo "[${DONE}/${TOTAL}] rendering..."
    if ! run_render "$model_dir" "$log_path"; then
      echo "[warn] render failed: $model_dir" >&2
      FAILURES+=("${model_dir} [render]")
      continue
    fi
  fi

  if [ "$RUN_METRICS" = "1" ]; then
    echo "[${DONE}/${TOTAL}] evaluating..."
    if ! run_metrics "$model_dir" "$source_path" "$air_source_path" "$log_path"; then
      echo "[warn] metrics failed: $model_dir" >&2
      FAILURES+=("${model_dir} [metrics]")
      continue
    fi
  fi

  echo "[${DONE}/${TOTAL}] done"
done

echo "[info] logs saved under: $LOG_ROOT"

if [ "${#FAILURES[@]}" -gt 0 ]; then
  echo "[fatal] ${#FAILURES[@]} runs failed:" >&2
  for item in "${FAILURES[@]}"; do
    echo "  - $item" >&2
  done
  exit 1
fi

echo "[done] all exp2 render/eval jobs finished successfully"
