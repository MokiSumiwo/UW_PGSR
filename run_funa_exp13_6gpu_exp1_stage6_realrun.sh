#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DATA_ROOT="${DATA_ROOT:-/mnt/d/home/ycy}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-${REPO_DIR}/outputs/funa_exp13_6gpu}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT_BASE}/logs/exp1_stage6_realrun}"
GPU_ID="${GPU_ID:-9}"
RENDER_ITERATION="${RENDER_ITERATION:-30000}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_TAG="${RUN_TAG:-_realrun}"

DEFAULT_SCENES=(
  Bay_submerged_ships_2
  Egypt_Ruins_Environment_2
  Eternal_temple_3
  oceanEnvironmentPark_2
  V3_Cave02
  V3_Dive3
)

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter: $PYTHON_BIN" >&2
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

scene_pseudo_path() {
  local scene_name="$1"
  local pseudo_path="${DATA_ROOT}/${scene_name}/with/blue/4/images/fake_air_Deep_Sea-NN"
  if [ -d "$pseudo_path" ]; then
    printf '%s\n' "$pseudo_path"
  else
    printf '%s\n' ""
  fi
}

scene_tag() {
  local scene_name="$1"
  printf '%s_blue_4\n' "$scene_name"
}

reference_stage6_dir() {
  local scene_name="$1"
  printf '%s/%s/exp1/%s_progressive_stage6_pseudoclean\n' \
    "$OUTPUT_ROOT_BASE" \
    "$scene_name" \
    "$(scene_tag "$scene_name")"
}

realrun_stage6_dir() {
  local scene_name="$1"
  printf '%s/%s/exp1/%s_progressive_stage6_pseudoclean%s\n' \
    "$OUTPUT_ROOT_BASE" \
    "$scene_name" \
    "$(scene_tag "$scene_name")" \
    "$RUN_TAG"
}

stage6_overrides_json() {
  cat <<'EOF'
{"use_underwater_model": true, "physics_close_background": 1, "physics_depth_gated_background": 1, "physics_far_to_binf": 1, "densify_block_far": 1, "lambda_bg_consistency": 0.002, "bg_hit_alpha_weight": 0.04, "bg_prune_enable": true, "lambda_fake_air": 0.02, "lambda_fake_air_lap": 0.05, "lambda_fake_air_lpips": 0.01}
EOF
}

run_nohup_cmd() {
  nohup "$@" &
  local pid=$!
  wait "$pid"
}

run_train_with_overrides() {
  local scene_path="$1"
  local model_path="$2"
  local air_path="$3"
  local fake_air_path="$4"
  local overrides_json="$5"

  OVERRIDES_JSON="$overrides_json" \
  SCENE_PATH="$scene_path" \
  MODEL_PATH="$model_path" \
  AIR_PATH="$air_path" \
  FAKE_AIR_PATH="$fake_air_path" \
  nohup "$PYTHON_BIN" - <<'PY' &
import json
import os
from argparse import ArgumentParser, Namespace

import torch

import train as train_mod
from arguments import ModelParams, OptimizationParams, PipelineParams

scene_path = os.environ["SCENE_PATH"]
model_path = os.environ["MODEL_PATH"]
air_path = os.environ.get("AIR_PATH", "")
fake_air_path = os.environ.get("FAKE_AIR_PATH", "")
overrides = json.loads(os.environ["OVERRIDES_JSON"])

parser = ArgumentParser(description="Exp1 stage6 realrun wrapper")
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
if fake_air_path:
    argv.extend(["--fake_air_source_path", fake_air_path])

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
  local pid=$!
  wait "$pid"
}

write_manifest() {
  local model_path="$1"
  local scene_path="$2"
  local air_path="$3"
  local fake_air_path="$4"
  local overrides_json="$5"
  local ref_dir="$6"

  MANIFEST_MODEL_PATH="$model_path" \
  MANIFEST_SCENE_PATH="$scene_path" \
  MANIFEST_AIR_PATH="$air_path" \
  MANIFEST_FAKE_AIR_PATH="$fake_air_path" \
  MANIFEST_OVERRIDES="$overrides_json" \
  MANIFEST_REF_DIR="$ref_dir" \
  MANIFEST_RUN_TAG="$RUN_TAG" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

model_path = Path(os.environ["MANIFEST_MODEL_PATH"])
payload = {
    "experiment_profile": "progressive_stage6_pseudoclean",
    "profile_family": "progressive",
    "profile_stage": 6,
    "profile_variant": "realrun",
    "run_tag": os.environ.get("MANIFEST_RUN_TAG", ""),
    "source_path": os.environ["MANIFEST_SCENE_PATH"],
    "air_source_path": os.environ.get("MANIFEST_AIR_PATH", ""),
    "fake_air_source_path": os.environ.get("MANIFEST_FAKE_AIR_PATH", ""),
    "reference_stage6_dir": os.environ.get("MANIFEST_REF_DIR", ""),
    "overrides": json.loads(os.environ["MANIFEST_OVERRIDES"]),
}
with open(model_path / "experiment_manifest.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

run_render_and_metrics() {
  local model_path="$1"
  local scene_path="$2"
  local air_path="$3"

  run_nohup_cmd "$PYTHON_BIN" render.py \
    --model_path "$model_path" \
    --iteration "$RENDER_ITERATION" \
    --skip_train \
    --air_bg_black 1 \
    --air_bg_use_hit_map 1

  if [ -n "$air_path" ] && [ -d "$air_path" ]; then
    run_nohup_cmd "$PYTHON_BIN" metrics3.py -m "$model_path" -s "$scene_path" -sa "$air_path"
  else
    run_nohup_cmd "$PYTHON_BIN" metrics3.py -m "$model_path" -s "$scene_path"
  fi
}

aggregate_outputs() {
  local -a model_paths=()
  local scene_name
  for scene_name in "${SCENES[@]}"; do
    local out_dir
    out_dir="$(realrun_stage6_dir "$scene_name")"
    if [ -f "${out_dir}/results.json" ] || [ -L "${out_dir}/results.json" ]; then
      model_paths+=("$out_dir")
    fi
  done

  if [ "${#model_paths[@]}" -eq 0 ]; then
    echo "[warn] no stage6 realrun results found, skip summary"
    return 0
  fi

  run_nohup_cmd "$PYTHON_BIN" scripts/analyze_experiments.py \
    -m "${model_paths[@]}" \
    --analysis progressive \
    --output_json "${OUTPUT_ROOT_BASE}/exp1_stage6_realrun_summary.json"
}

SCENES=()
if [ "$#" -gt 0 ]; then
  SCENES=("$@")
else
  SCENES=("${DEFAULT_SCENES[@]}")
fi

LOG_PATH="${LOG_ROOT}/stage6_realrun_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[info] log_path=$LOG_PATH"
echo "[info] gpu_id=$GPU_ID"
echo "[info] run_tag=$RUN_TAG"
echo "[info] scenes=${SCENES[*]}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

for scene_name in "${SCENES[@]}"; do
  scene_path="$(scene_source_path "$scene_name")"
  air_path="$(scene_air_path "$scene_name")"
  fake_air_path="$(scene_pseudo_path "$scene_name")"
  ref_dir="$(reference_stage6_dir "$scene_name")"
  out_dir="$(realrun_stage6_dir "$scene_name")"
  overrides_json="$(stage6_overrides_json)"

  if [ ! -d "$scene_path" ]; then
    echo "[fatal][${scene_name}] missing scene path: $scene_path" >&2
    exit 1
  fi

  if [ ! -d "$fake_air_path" ]; then
    echo "[fatal][${scene_name}] missing pseudo-clean dir: $fake_air_path" >&2
    exit 1
  fi

  if [ ! -d "$ref_dir" ]; then
    echo "[warn][${scene_name}] canonical reference dir not found: $ref_dir"
  else
    echo "[info][${scene_name}] preserve existing reference dir: $ref_dir"
  fi

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "${out_dir}/results.json" ]; then
    echo "[skip][${scene_name}] existing results: ${out_dir}/results.json"
    continue
  fi

  echo "[train][${scene_name}] out=$out_dir"
  run_train_with_overrides "$scene_path" "$out_dir" "$air_path" "$fake_air_path" "$overrides_json"
  write_manifest "$out_dir" "$scene_path" "$air_path" "$fake_air_path" "$overrides_json" "$ref_dir"

  echo "[eval][${scene_name}] render+metrics"
  run_render_and_metrics "$out_dir" "$scene_path" "$air_path"
done

aggregate_outputs

echo "[done] exp1 stage6 realrun finished on gpu ${GPU_ID}"
