#!/usr/bin/env python3

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


OUTPUT_ROOT = Path("/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu")

EXP1_STAGE_ORDER = [
    "stage0_pgsr",
    "stage1_underwater",
    "stage2_bgclosure",
    "stage3_fargating",
    "stage4_capacity",
    "stage5_conflict",
    "stage6_pseudoclean",
]

EXP1_STAGE_LABELS = {
    "stage0_pgsr": "Stage 0: PGSR",
    "stage1_underwater": "Stage 1: + Underwater",
    "stage2_bgclosure": "Stage 2: + BG Closure",
    "stage3_fargating": "Stage 3: + Far Gating",
    "stage4_capacity": "Stage 4: + Capacity Control",
    "stage5_conflict": "Stage 5: + Conflict-Aware BG",
    "stage6_pseudoclean": "Stage 6: + Pseudo-Clean",
}

METRIC_MAP = {
    "rgb_ssim": "SSIM",
    "rgb_psnr": "PSNR",
    "rgb_lpips": "LPIPS",
    "clean_ssim": "AIRBGBLACK_SSIM",
    "clean_psnr": "AIRBGBLACK_PSNR",
    "clean_lpips": "AIRBGBLACK_LPIPS",
    "depth_rmse": "depth.rmse",
    "depth_absrel": "depth.absrel",
    "depth_delta125": "depth.delta<1.25",
    "normal_mean_angle": "normal.mean_angle_deg",
    "normal_median_angle": "normal.median_angle_deg",
    "normal_acc_11_25deg": "normal.acc@11.25deg",
}


def load_results(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["ours_30000"]


def metric_row(metrics: dict) -> dict:
    row = {}
    for out_key, src_key in METRIC_MAP.items():
        row[out_key] = float(metrics[src_key])
    return row


def mean_rows(rows):
    merged = {}
    for out_key in METRIC_MAP:
        merged[out_key] = sum(r[out_key] for r in rows) / len(rows)
    return merged


def make_avg_row(rows: list[dict], extra: dict) -> dict:
    return {
        **extra,
        "scope": "avg",
        "scene": "ALL",
        "num_scenes": len(rows),
        **mean_rows(rows),
    }


def make_scene_row(scene: str, metrics: dict, extra: dict) -> dict:
    return {
        **extra,
        "scope": "scene",
        "scene": scene,
        "num_scenes": 1,
        **metric_row(metrics),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_exp1_rows() -> list[dict]:
    rows = []
    scenes = sorted([p.name for p in OUTPUT_ROOT.iterdir() if p.is_dir() and (p / "exp1").is_dir()])
    stage_records = {}

    for stage in EXP1_STAGE_ORDER:
        stage_rows = []
        stage_variant = "single"
        scene_rows = {}
        for scene in scenes:
            if stage != "stage6_pseudoclean":
                result_path = next(
                    (OUTPUT_ROOT / scene / "exp1").glob(f"*progressive_{stage}/results.json"),
                    None,
                )
            else:
                realrun_path = next(
                    (OUTPUT_ROOT / scene / "exp1").glob("*progressive_stage6_pseudoclean_realrun/results.json"),
                    None,
                )
                if realrun_path is not None:
                    result_path = realrun_path
                    stage_variant = "realrun"
                else:
                    result_path = OUTPUT_ROOT / scene / "exp2" / f"{scene}_blue_4" / "results.json"
                    stage_variant = "orig_ref"

            metrics = load_results(result_path)
            stage_rows.append(metric_row(metrics))
            scene_rows[scene] = metrics

        extra = {
            "stage": stage,
            "stage_label": EXP1_STAGE_LABELS[stage],
            "variant": stage_variant,
        }

        stage_records[stage] = {
            "avg": make_avg_row(stage_rows, extra),
            "scenes": {
                scene: make_scene_row(scene, metrics, extra) for scene, metrics in scene_rows.items()
            },
        }

    for scene in ["ALL"] + scenes:
        for stage in EXP1_STAGE_ORDER:
            if scene == "ALL":
                rows.append(stage_records[stage]["avg"])
            else:
                rows.append(stage_records[stage]["scenes"][scene])

    return rows


def collect_exp2_rows() -> list[dict]:
    buckets = defaultdict(list)
    scenes = set()
    for results_path in sorted(OUTPUT_ROOT.glob("*/exp2/*/results.json")):
        scene = results_path.parts[-4]
        scenes.add(scene)
        model_name = results_path.parent.name
        match = re.match(r"(.+)_(blue|green)_(\d+)$", model_name)
        if match is None:
            continue
        color = match.group(2)
        level = int(match.group(3))
        buckets[(color, level)].append((scene, load_results(results_path)))

    rows = []
    config_records = {}
    for color in ("blue", "green"):
        for level in (1, 2, 3, 4):
            key = (color, level)
            if key not in buckets:
                continue
            bucket_rows = [metric_row(metrics) for _, metrics in buckets[key]]
            extra = {
                "water_color": color,
                "degradation_level": level,
            }
            config_records[key] = {
                "avg": make_avg_row(bucket_rows, extra),
                "scenes": {
                    scene: make_scene_row(scene, metrics, extra)
                    for scene, metrics in sorted(buckets[key])
                },
            }

    for scene in ["ALL"] + sorted(scenes):
        for color in ("blue", "green"):
            for level in (1, 2, 3, 4):
                key = (color, level)
                if key not in config_records:
                    continue
                if scene == "ALL":
                    rows.append(config_records[key]["avg"])
                else:
                    rows.append(config_records[key]["scenes"][scene])
    return rows


def collect_exp3_rows() -> list[dict]:
    buckets = defaultdict(list)
    scenes = set()
    for results_path in sorted(OUTPUT_ROOT.glob("*/exp3/*/results.json")):
        scene = results_path.parts[-4]
        scenes.add(scene)
        model_name = results_path.parent.name
        if model_name.endswith("fakeair_none"):
            profile = "fakeair_none"
        elif model_name.endswith("fakeair_pseudo"):
            profile = "fakeair_pseudo"
        elif model_name.endswith("fakeair_oracle"):
            profile = "fakeair_oracle"
        else:
            continue
        buckets[profile].append((scene, load_results(results_path)))

    order = ["fakeair_none", "fakeair_pseudo", "fakeair_oracle"]
    rows = []
    profile_records = {}
    for profile in order:
        if profile not in buckets:
            continue
        bucket_rows = [metric_row(metrics) for _, metrics in buckets[profile]]
        extra = {"profile": profile}
        profile_records[profile] = {
            "avg": make_avg_row(bucket_rows, extra),
            "scenes": {
                scene: make_scene_row(scene, metrics, extra)
                for scene, metrics in sorted(buckets[profile])
            },
        }

    for scene in ["ALL"] + sorted(scenes):
        for profile in order:
            if profile not in profile_records:
                continue
            if scene == "ALL":
                rows.append(profile_records[profile]["avg"])
            else:
                rows.append(profile_records[profile]["scenes"][scene])
    return rows


def main():
    exp1_rows = collect_exp1_rows()
    exp2_rows = collect_exp2_rows()
    exp3_rows = collect_exp3_rows()

    common_fields = list(METRIC_MAP.keys())

    write_csv(
        OUTPUT_ROOT / "exp1_table.csv",
        exp1_rows,
        ["scope", "scene", "stage", "stage_label", "variant", "num_scenes"] + common_fields,
    )
    write_csv(
        OUTPUT_ROOT / "exp2_table.csv",
        exp2_rows,
        ["scope", "scene", "water_color", "degradation_level", "num_scenes"] + common_fields,
    )
    write_csv(
        OUTPUT_ROOT / "exp3_table.csv",
        exp3_rows,
        ["scope", "scene", "profile", "num_scenes"] + common_fields,
    )

    print(OUTPUT_ROOT / "exp1_table.csv")
    print(OUTPUT_ROOT / "exp2_table.csv")
    print(OUTPUT_ROOT / "exp3_table.csv")


if __name__ == "__main__":
    main()
