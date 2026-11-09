#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DATA_ROOT="${DATA_ROOT:-/mnt/d/home/ycy}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-${REPO_DIR}/outputs/funa_exp13_6gpu}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT_BASE}/logs/exp1_remaining}"
RENDER_ITERATION="${RENDER_ITERATION:-30000}"
REGISTER_FULL_REF="${REGISTER_FULL_REF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SCENES=(
  Bay_submerged_ships_2
  Egypt_Ruins_Environment_2
  Eternal_temple_3
  oceanEnvironmentPark_2
  V3_Cave02
  V3_Dive3
)

GPUS=(3 4 5 6 7 8)

STAGES=(
  progressive_stage0_pgsr
  progressive_stage1_underwater
  progressive_stage2_bgclosure
  progressive_stage3_fargating
  progressive_stage4_capacity
  progressive_stage5_conflict
)

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

if [ "${#SCENES[@]}" -ne "${#GPUS[@]}" ]; then
  echo "SCENES and GPUS must have the same length." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT_BASE" "$LOG_ROOT"

scene_source_path() {
  local scene_name="$1"
  printf '%s/%s/with/blue/4\n' "$DATA_ROOT" "$scene_name"
}

scene_air_path() {
  local scene_name="$1"
  local air_path="${DATA_ROOT}/${scene_name}/with/air"
  if [ -d "$air_path" ]; then
    printf '%s\n' "$air_path"
  else
    printf '%s\n' ""
  fi
}

scene_tag() {
  local scene_name="$1"
  printf '%s_blue_4\n' "$scene_name"
}

stage_output_dir() {
  local scene_name="$1"
  local stage_name="$2"
  printf '%s/%s/exp1/%s_%s\n' \
    "$OUTPUT_ROOT_BASE" \
    "$scene_name" \
    "$(scene_tag "$scene_name")" \
    "$stage_name"
}

full_model_dir() {
  local scene_name="$1"
  printf '%s/%s/exp2/%s\n' \
    "$OUTPUT_ROOT_BASE" \
    "$scene_name" \
    "$(scene_tag "$scene_name")"
}

stage_overrides_json() {
  local stage_name="$1"
  case "$stage_name" in
    progressive_stage0_pgsr)
      cat <<'EOF'
{"use_underwater_model": false, "physics_close_background": 0, "physics_depth_gated_background": 0, "physics_far_to_binf": 0, "densify_block_far": 0, "lambda_bg_consistency": 0.0, "bg_hit_alpha_weight": 0.0, "bg_prune_enable": false, "lambda_fake_air": 0.0, "lambda_fake_air_lap": 0.0, "lambda_fake_air_lpips": 0.0}
EOF
      ;;
    progressive_stage1_underwater)
      cat <<'EOF'
{"use_underwater_model": true, "physics_close_background": 0, "physics_depth_gated_background": 0, "physics_far_to_binf": 0, "densify_block_far": 0, "lambda_bg_consistency": 0.0, "bg_hit_alpha_weight": 0.0, "bg_prune_enable": false, "lambda_fake_air": 0.0, "lambda_fake_air_lap": 0.0, "lambda_fake_air_lpips": 0.0}
EOF
      ;;
    progressive_stage2_bgclosure)
      cat <<'EOF'
{"use_underwater_model": true, "physics_close_background": 1, "physics_depth_gated_background": 0, "physics_far_to_binf": 0, "densify_block_far": 0, "lambda_bg_consistency": 0.0, "bg_hit_alpha_weight": 0.0, "bg_prune_enable": false, "lambda_fake_air": 0.0, "lambda_fake_air_lap": 0.0, "lambda_fake_air_lpips": 0.0}
EOF
      ;;
    progressive_stage3_fargating)
      cat <<'EOF'
{"use_underwater_model": true, "physics_close_background": 1, "physics_depth_gated_background": 1, "physics_far_to_binf": 1, "densify_block_far": 0, "lambda_bg_consistency": 0.0, "bg_hit_alpha_weight": 0.0, "bg_prune_enable": false, "lambda_fake_air": 0.0, "lambda_fake_air_lap": 0.0, "lambda_fake_air_lpips": 0.0}
EOF
      ;;
    progressive_stage4_capacity)
      cat <<'EOF'
{"use_underwater_model": true, "physics_close_background": 1, "physics_depth_gated_background": 1, "physics_far_to_binf": 1, "densify_block_far": 1, "lambda_bg_consistency": 0.0, "bg_hit_alpha_weight": 0.0, "bg_prune_enable": false, "lambda_fake_air": 0.0, "lambda_fake_air_lap": 0.0, "lambda_fake_air_lpips": 0.0}
EOF
      ;;
    progressive_stage5_conflict)
      cat <<'EOF'
{"use_underwater_model": true, "physics_close_background": 1, "physics_depth_gated_background": 1, "physics_far_to_binf": 1, "densify_block_far": 1, "lambda_bg_consistency": 0.002, "bg_hit_alpha_weight": 0.04, "bg_prune_enable": true, "lambda_fake_air": 0.0, "lambda_fake_air_lap": 0.0, "lambda_fake_air_lpips": 0.0}
EOF
      ;;
    *)
      echo "Unknown stage: $stage_name" >&2
      return 1
      ;;
  esac
}

run_train_with_overrides() {
  local scene_path="$1"
  local model_path="$2"
  local air_path="$3"
  local stage_name="$4"
  local overrides_json="$5"

  OVERRIDES_JSON="$overrides_json" \
  SCENE_PATH="$scene_path" \
  MODEL_PATH="$model_path" \
  AIR_PATH="$air_path" \
  STAGE_NAME="$stage_name" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from argparse import ArgumentParser, Namespace

import torch

import train as train_mod
from arguments import ModelParams, OptimizationParams, PipelineParams

scene_path = os.environ["SCENE_PATH"]
model_path = os.environ["MODEL_PATH"]
air_path = os.environ.get("AIR_PATH", "")
stage_name = os.environ["STAGE_NAME"]
overrides = json.loads(os.environ["OVERRIDES_JSON"])

parser = ArgumentParser(description="Profiled training wrapper")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6007)
parser.add_argument('--debug_from', type=int, default=-100)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--deterministic', type=int, default=0)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default=None)

argv = [
    "-s", scene_path,
    "--model_path", model_path,
    "--eval",
]
if air_path:
    argv.extend(["--air_source_path", air_path])

args = parser.parse_args(argv)
for key, value in overrides.items():
    setattr(args, key, value)

if args.iterations not in args.save_iterations:
    args.save_iterations.append(args.iterations)

os.makedirs(args.model_path, exist_ok=True)

lp_extracted = lp.extract(args)
op_extracted = op.extract(args)
pp_extracted = pp.extract(args)

merged_vars = vars(args).copy()
merged_vars.update(vars(lp_extracted))
merged_vars.update(vars(op_extracted))
merged_vars.update(vars(pp_extracted))

with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
    f.write(str(Namespace(**merged_vars)))

train_mod.setup_seed(int(getattr(args, "seed", 23)))
train_mod._maybe_enable_determinism(bool(int(getattr(args, "deterministic", 0)) == 1))
train_mod.safe_state(args.quiet)
torch.autograd.set_detect_anomaly(args.detect_anomaly)

train_mod.training(
    lp_extracted,
    op_extracted,
    pp_extracted,
    args.test_iterations,
    args.save_iterations,
    args.checkpoint_iterations,
    args.start_checkpoint,
    args.debug_from,
)
PY
}

write_manifest() {
  local model_path="$1"
  local stage_name="$2"
  local stage_idx="$3"
  local scene_path="$4"
  local air_path="$5"
  local overrides_json="$6"

  MANIFEST_MODEL_PATH="$model_path" \
  MANIFEST_STAGE_NAME="$stage_name" \
  MANIFEST_STAGE_IDX="$stage_idx" \
  MANIFEST_SCENE_PATH="$scene_path" \
  MANIFEST_AIR_PATH="$air_path" \
  MANIFEST_OVERRIDES="$overrides_json" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

model_path = Path(os.environ["MANIFEST_MODEL_PATH"])
payload = {
    "experiment_profile": os.environ["MANIFEST_STAGE_NAME"],
    "profile_family": "progressive",
    "profile_stage": int(os.environ["MANIFEST_STAGE_IDX"]),
    "source_path": os.environ["MANIFEST_SCENE_PATH"],
    "air_source_path": os.environ.get("MANIFEST_AIR_PATH", ""),
    "overrides": json.loads(os.environ["MANIFEST_OVERRIDES"]),
}
with open(model_path / "experiment_manifest.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

register_full_reference() {
  local scene_name="$1"
  local stage_name="progressive_stage6_pseudoclean"
  local ref_src
  local ref_dst
  local scene_path
  local air_path

  ref_src="$(full_model_dir "$scene_name")"
  ref_dst="$(stage_output_dir "$scene_name" "$stage_name")"
  scene_path="$(scene_source_path "$scene_name")"
  air_path="$(scene_air_path "$scene_name")"

  if [ ! -d "$ref_src" ]; then
    echo "[warn][${scene_name}] full reference missing, skip stage6 registration: $ref_src" >&2
    return 0
  fi

  mkdir -p "$ref_dst"

  for item in results.json scene_paths.json cfg_args test point_cloud; do
    if [ -e "${ref_src}/${item}" ] && [ ! -e "${ref_dst}/${item}" ]; then
      ln -s "${ref_src}/${item}" "${ref_dst}/${item}"
    fi
  done

  MANIFEST_MODEL_PATH="$ref_dst" \
  MANIFEST_STAGE_NAME="$stage_name" \
  MANIFEST_STAGE_IDX="6" \
  MANIFEST_SCENE_PATH="$scene_path" \
  MANIFEST_AIR_PATH="$air_path" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

model_path = Path(os.environ["MANIFEST_MODEL_PATH"])
payload = {
    "experiment_profile": os.environ["MANIFEST_STAGE_NAME"],
    "profile_family": "progressive",
    "profile_stage": int(os.environ["MANIFEST_STAGE_IDX"]),
    "source_path": os.environ["MANIFEST_SCENE_PATH"],
    "air_source_path": os.environ.get("MANIFEST_AIR_PATH", ""),
    "reference_only": True,
    "reference_source": "exp2_full_model",
}
with open(model_path / "experiment_manifest.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

run_render_and_metrics() {
  local model_path="$1"
  local scene_path="$2"
  local air_path="$3"

  "$PYTHON_BIN" render.py \
    --model_path "$model_path" \
    --iteration "$RENDER_ITERATION" \
    --skip_train \
    --air_bg_black 1 \
    --air_bg_use_hit_map 1

  if [ -n "$air_path" ] && [ -d "$air_path" ]; then
    "$PYTHON_BIN" metrics3.py -m "$model_path" -s "$scene_path" -sa "$air_path"
  else
    "$PYTHON_BIN" metrics3.py -m "$model_path" -s "$scene_path"
  fi
}

launch_scene_worker() {
  local scene_name="$1"
  local gpu_id="$2"
  local scene_path
  local air_path
  local log_path="${LOG_ROOT}/${scene_name}.log"
  local worker_cmd

  scene_path="$(scene_source_path "$scene_name")"
  air_path="$(scene_air_path "$scene_name")"

  if [ ! -d "$scene_path" ]; then
    echo "[fatal] missing scene path: $scene_path" >&2
    return 1
  fi

  worker_cmd="$(cat <<EOF
$(declare -f scene_source_path)
$(declare -f scene_air_path)
$(declare -f scene_tag)
$(declare -f stage_output_dir)
$(declare -f full_model_dir)
$(declare -f stage_overrides_json)
$(declare -f run_train_with_overrides)
$(declare -f write_manifest)
$(declare -f register_full_reference)
$(declare -f run_render_and_metrics)
set -euo pipefail
cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHON_BIN="$PYTHON_BIN"
export REPO_DIR="$REPO_DIR"
export OUTPUT_ROOT_BASE="$OUTPUT_ROOT_BASE"
export RENDER_ITERATION="$RENDER_ITERATION"
export REGISTER_FULL_REF="$REGISTER_FULL_REF"
export SKIP_EXISTING="$SKIP_EXISTING"
export DATA_ROOT="$DATA_ROOT"
scene_name="$scene_name"
scene_path="$scene_path"
air_path="$air_path"
stage_list=(
  progressive_stage0_pgsr
  progressive_stage1_underwater
  progressive_stage2_bgclosure
  progressive_stage3_fargating
  progressive_stage4_capacity
  progressive_stage5_conflict
)
if [ "\$REGISTER_FULL_REF" = "1" ]; then
  register_full_reference "\$scene_name"
fi
for idx in "\${!stage_list[@]}"; do
  stage_name="\${stage_list[\$idx]}"
  out_dir="\$(stage_output_dir "\$scene_name" "\$stage_name")"
  overrides_json="\$(stage_overrides_json "\$stage_name")"
  if [ "\$SKIP_EXISTING" = "1" ] && [ -f "\${out_dir}/results.json" ]; then
    echo "[worker][\${scene_name}] skip existing stage=\${stage_name} out=\${out_dir}"
    write_manifest "\$out_dir" "\$stage_name" "\$idx" "\$scene_path" "\$air_path" "\$overrides_json"
    continue
  fi
  echo "[worker][\${scene_name}] gpu=\${CUDA_VISIBLE_DEVICES} stage=\${stage_name} out=\${out_dir}"
  run_train_with_overrides "\$scene_path" "\$out_dir" "\$air_path" "\$stage_name" "\$overrides_json"
  write_manifest "\$out_dir" "\$stage_name" "\$idx" "\$scene_path" "\$air_path" "\$overrides_json"
  run_render_and_metrics "\$out_dir" "\$scene_path" "\$air_path"
done
echo "[worker][\${scene_name}] finished"
EOF
)"

  nohup bash -lc "$worker_cmd" >"$log_path" 2>&1 &

  PIDS+=("$!")
}

aggregate_outputs() {
  local -a model_paths=()
  while IFS= read -r path; do
    model_paths+=("$path")
  done < <(find "$OUTPUT_ROOT_BASE" \( -type f -o -type l \) -path '*/exp1/*/results.json' -printf '%h\n' | sort)

  if [ "${#model_paths[@]}" -eq 0 ]; then
    echo "[warn] no exp1 results found, skip aggregate summary"
    return 0
  fi

  "$PYTHON_BIN" scripts/analyze_experiments.py \
    -m "${model_paths[@]}" \
    --analysis progressive \
    --output_json "${OUTPUT_ROOT_BASE}/exp1_all_progressive_summary.json"

  "$PYTHON_BIN" scripts/export_exp1_progressive_paper_table.py \
    --root "$OUTPUT_ROOT_BASE" \
    --output_dir "$OUTPUT_ROOT_BASE"
}

PIDS=()
for idx in "${!SCENES[@]}"; do
  launch_scene_worker "${SCENES[$idx]}" "${GPUS[$idx]}"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "[fatal] at least one worker failed. Check logs under $LOG_ROOT" >&2
  exit 1
fi

aggregate_outputs

echo "[done] exp1 remaining stages finished"
echo "[done] logs: $LOG_ROOT"
echo "[done] outputs: $OUTPUT_ROOT_BASE"
