#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


PROGRESSIVE_ORDER = {
    "progressive_stage0_pgsr": 0,
    "progressive_stage1_underwater": 1,
    "progressive_stage2_bgclosure": 2,
    "progressive_stage3_fargating": 3,
    "progressive_stage4_capacity": 4,
    "progressive_stage5_conflict": 5,
    "progressive_stage6_pseudoclean": 6,
    "progressive_full": 6,
}

FAKEAIR_ORDER = {
    "none": 0,
    "fake_air": 1,
    "oracle": 2,
}

DEFAULT_METRICS = [
    "SSIM",
    "PSNR",
    "AIR_PSNR",
    "depth.rmse",
    "normal.mean_angle_deg",
    "test_far_alpha_mass",
]


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if math.isfinite(out):
        return out
    return float("nan")


def _nanmean(values: List[float]) -> float:
    vals = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def _pick_method(results: Dict[str, dict], requested: Optional[str]) -> Tuple[str, dict]:
    if not results:
        raise ValueError("results.json is empty")
    if requested:
        if requested not in results:
            raise KeyError(f"Method {requested} not found in results.json")
        return requested, results[requested]
    if len(results) == 1:
        key = next(iter(results.keys()))
        return key, results[key]

    ours = sorted(
        [k for k in results.keys() if k.startswith("ours_")],
        key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1,
    )
    if ours:
        key = ours[-1]
        return key, results[key]

    key = sorted(results.keys())[0]
    return key, results[key]


def _scene_name_from_source(source_path: str, fallback: str) -> str:
    if not source_path:
        return fallback
    parts = Path(source_path).parts
    for idx, token in enumerate(parts):
        if token == "with" and idx > 0:
            return parts[idx - 1]
    return Path(source_path).name or fallback


def load_record(model_path: Path, method: Optional[str]) -> Optional[dict]:
    results_path = model_path / "results.json"
    if not results_path.exists():
        print(f"[skip] missing results.json: {model_path}")
        return None

    results = _load_json(results_path)
    method_name, metrics = _pick_method(results, method)

    scene_paths = _load_json(model_path / "scene_paths.json") if (model_path / "scene_paths.json").exists() else {}
    manifest = _load_json(model_path / "experiment_manifest.json") if (model_path / "experiment_manifest.json").exists() else {}
    eval_summary = _load_json(model_path / "experiment_eval.json") if (model_path / "experiment_eval.json").exists() else {}

    latest_test = eval_summary.get("latest", {}).get("test", {})
    source_path = scene_paths.get("source_path", manifest.get("source_path", ""))

    record = {
        "model_path": str(model_path),
        "scene_name": _scene_name_from_source(source_path, model_path.name),
        "method": method_name,
        "experiment_profile": manifest.get("experiment_profile", scene_paths.get("experiment_profile", "")),
        "profile_family": manifest.get("profile_family", "custom"),
        "profile_stage": manifest.get("profile_stage", None),
        "air_supervision_source": manifest.get("air_supervision_source", scene_paths.get("air_supervision_source", "fake_air")),
        "source_path": source_path,
        "air_source_path": scene_paths.get("air_source_path", manifest.get("air_source_path", "")),
        "fake_air_source_path": scene_paths.get("fake_air_source_path", manifest.get("fake_air_source_path", "")),
        "test_far_alpha_mass": _safe_float(latest_test.get("far_alpha_mass", float("nan"))),
    }

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            record[key] = _safe_float(value)
        else:
            record[key] = value
    return record


def summarize(records: List[dict], analysis: str, metrics: List[str]) -> List[dict]:
    groups: Dict[str, dict] = {}

    for record in records:
        if analysis == "progressive":
            label = record.get("experiment_profile", "") or "custom"
            order = record.get("profile_stage", PROGRESSIVE_ORDER.get(label, 999))
        elif analysis == "fakeair":
            label = str(record.get("air_supervision_source", "unknown")).lower()
            order = FAKEAIR_ORDER.get(label, 999)
        else:
            label = record.get("experiment_profile", "") or record.get("scene_name", "unknown")
            order = 999

        bucket = groups.setdefault(label, {"label": label, "order": order, "records": []})
        bucket["records"].append(record)

    rows = []
    for bucket in groups.values():
        row = {
            "group": bucket["label"],
            "num_runs": len(bucket["records"]),
        }
        for metric in metrics:
            row[metric] = _nanmean([r.get(metric, float("nan")) for r in bucket["records"]])
        rows.append((bucket["order"], bucket["label"], row))

    rows.sort(key=lambda x: (x[0], x[1]))
    return [row for _, _, row in rows]


def write_csv(path: Path, rows: List[dict], metrics: List[str]):
    fieldnames = ["group", "num_runs"] + metrics
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize progressive ablation or fake-air bias experiments.")
    parser.add_argument("--model_paths", "-m", nargs="+", required=True, help="List of scene output directories.")
    parser.add_argument("--analysis", choices=["progressive", "fakeair", "profiles"], default="progressive")
    parser.add_argument("--method", default=None, help="Method key inside results.json. Default: auto-pick.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    records = []
    for model_path in args.model_paths:
        rec = load_record(Path(model_path), args.method)
        if rec is not None:
            records.append(rec)

    if not records:
        raise SystemExit("No valid experiment records found.")

    rows = summarize(records, args.analysis, args.metrics)

    header = ["group", "num_runs"] + args.metrics
    print("\t".join(header))
    for row in rows:
        vals = [row.get("group", ""), str(row.get("num_runs", 0))]
        for metric in args.metrics:
            value = row.get(metric, float("nan"))
            vals.append(f"{value:.6f}" if isinstance(value, float) and math.isfinite(value) else "nan")
        print("\t".join(vals))

    if args.output_json:
        payload = {
            "analysis": args.analysis,
            "metrics": args.metrics,
            "rows": rows,
            "records": records,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.output_csv:
        write_csv(Path(args.output_csv), rows, args.metrics)


if __name__ == "__main__":
    main()
