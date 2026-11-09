#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import Imath
import OpenEXR


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_METRICS = ["clean_psnr", "rgb_ssim", "depth_rmse", "normal_mean_angle", "far_alpha_mass"]


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _nanmean(values: List[float]) -> float:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


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
        [k for k in results if k.startswith("ours_")],
        key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1,
    )
    if ours:
        key = ours[-1]
        return key, results[key]
    key = sorted(results.keys())[0]
    return key, results[key]


def read_exr_channel(file_path: str, channel_name: str = "R") -> np.ndarray:
    exr_file = OpenEXR.InputFile(file_path)
    try:
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = list(header["channels"].keys())
        ch = channel_name if channel_name in channels else None
        if ch is None:
            lower_map = {c.lower(): c for c in channels}
            ch = lower_map.get(channel_name.lower(), None)
        if ch is None:
            raise KeyError(f"Channel {channel_name} not found in {file_path}")
        try:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            buf = exr_file.channel(ch, pt)
            arr = np.frombuffer(buf, dtype=np.float32)
        except Exception:
            pt = Imath.PixelType(Imath.PixelType.HALF)
            buf = exr_file.channel(ch, pt)
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        return arr.reshape((height, width))
    finally:
        del exr_file


def _resolve_color_dir(base_path: Optional[str]) -> Optional[Path]:
    if base_path is None or not str(base_path).strip():
        return None
    p = Path(base_path)
    if (p / "images" / "ColorImage").is_dir():
        return p / "images" / "ColorImage"
    if p.is_dir():
        return p
    return None


def _infer_air_source_path_from_source(source_path: Optional[str]) -> Optional[Path]:
    if source_path is None or not str(source_path).strip():
        return None
    p = Path(source_path)
    try:
        air_dir = p.parent.parent / "air"
    except Exception:
        return None
    return air_dir if air_dir.is_dir() else None


def _parse_scene_tokens(source_path: str, fallback: str) -> dict:
    scene_name = fallback
    water_color = "unknown"
    degradation_level = "unknown"
    parts = Path(source_path).parts if source_path else ()
    for idx, token in enumerate(parts):
        if token == "with" and idx > 0 and idx + 2 < len(parts):
            scene_name = parts[idx - 1]
            water_color = parts[idx + 1]
            degradation_level = parts[idx + 2]
            break
    return {
        "scene_name": scene_name,
        "water_color": water_color,
        "degradation_level": degradation_level,
    }


def _iter_image_files(color_dir: Path):
    if color_dir is None or (not color_dir.exists()):
        return []
    return sorted([p for p in color_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def _gradient_strength(image_path: Path) -> float:
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    grad = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(grad))


def compute_scene_meta(source_path: str, air_source_path: Optional[str], void_raw: float, void_eps: float) -> dict:
    depth_dir = Path(source_path) / "images" / "DepthImage"
    depth_files = sorted(depth_dir.glob("*.exr")) if depth_dir.is_dir() else []

    per_img_p95 = []
    per_img_med = []
    per_img_bg_ratio = []
    for depth_path in depth_files:
        raw = read_exr_channel(str(depth_path), "R").astype(np.float32)
        valid = np.isfinite(raw) & (np.abs(raw - float(void_raw)) > float(void_eps))
        per_img_bg_ratio.append(float(1.0 - valid.mean()))
        if np.any(valid):
            depth_m = raw[valid] * 1000.0
            per_img_p95.append(float(np.quantile(depth_m, 0.95)))
            per_img_med.append(float(np.quantile(depth_m, 0.50)))

    air_dir = _resolve_color_dir(air_source_path)
    if air_dir is None:
        inferred_air = _infer_air_source_path_from_source(source_path)
        air_dir = _resolve_color_dir(str(inferred_air) if inferred_air is not None else None)
    color_dir = air_dir if air_dir is not None else _resolve_color_dir(source_path)
    img_files = _iter_image_files(color_dir) if color_dir is not None else []
    texture_vals = [_gradient_strength(p) for p in img_files]

    return {
        "depth_p95_m": _nanmean(per_img_p95),
        "depth_median_m": _nanmean(per_img_med),
        "background_ratio": _nanmean(per_img_bg_ratio),
        "texture_strength": _nanmean(texture_vals),
    }


def _pick_clean_psnr(metrics: dict) -> float:
    for key in ("AIR_PSNR", "AIRBGBLACK_PSNR"):
        if key in metrics:
            return _safe_float(metrics[key])
    return float("nan")


def load_scene_record(model_path: Path, method: Optional[str], scene_meta_cache: Dict[Tuple[str, str], dict], void_raw: float, void_eps: float) -> Optional[dict]:
    results_path = model_path / "results.json"
    scene_paths_path = model_path / "scene_paths.json"
    if (not results_path.exists()) or (not scene_paths_path.exists()):
        print(f"[skip] missing results.json or scene_paths.json: {model_path}")
        return None

    results = _load_json(results_path)
    method_name, metrics = _pick_method(results, method)
    scene_paths = _load_json(scene_paths_path)
    eval_summary = _load_json(model_path / "experiment_eval.json") if (model_path / "experiment_eval.json").exists() else {}

    source_path = scene_paths.get("source_path", "")
    air_source_path = scene_paths.get("air_source_path", "")
    cache_key = (source_path, air_source_path)
    if cache_key not in scene_meta_cache:
        scene_meta_cache[cache_key] = compute_scene_meta(source_path, air_source_path, void_raw, void_eps)
    scene_meta = scene_meta_cache[cache_key]
    tokens = _parse_scene_tokens(source_path, model_path.name)

    latest_test = eval_summary.get("latest", {}).get("test", {})
    record = {
        "model_path": str(model_path),
        "method": method_name,
        "scene_name": tokens["scene_name"],
        "water_color": tokens["water_color"],
        "degradation_level": str(tokens["degradation_level"]),
        "clean_psnr": _pick_clean_psnr(metrics),
        "rgb_ssim": _safe_float(metrics.get("SSIM", float("nan"))),
        "depth_rmse": _safe_float(metrics.get("depth.rmse", float("nan"))),
        "normal_mean_angle": _safe_float(metrics.get("normal.mean_angle_deg", float("nan"))),
        "far_alpha_mass": _safe_float(latest_test.get("far_alpha_mass", float("nan"))),
    }
    record.update(scene_meta)
    return record


def make_quantile_bins(records: List[dict], key: str, labels: List[str]) -> Tuple[List[float], List[str]]:
    values = np.asarray([_safe_float(r.get(key, float("nan"))) for r in records], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [], labels
    quantiles = np.linspace(0.0, 1.0, num=len(labels) + 1)[1:-1]
    edges = [float(np.quantile(values, q)) for q in quantiles]
    return edges, labels


def assign_bin(value: float, edges: List[float], labels: List[str]) -> str:
    if not math.isfinite(value):
        return "unknown"
    for idx, edge in enumerate(edges):
        if value <= edge:
            return labels[idx]
    return labels[-1]


def summarize_groups(records: List[dict], group_key: str, metrics: List[str]) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get(group_key, "unknown")), []).append(record)

    out = {}
    for group_name, group_records in sorted(groups.items(), key=lambda x: x[0]):
        row = {"num_scenes": len(group_records)}
        for metric in metrics:
            row[metric] = _nanmean([r.get(metric, float("nan")) for r in group_records])
        out[group_name] = row
    return out


def main():
    parser = argparse.ArgumentParser(description="Scene-stratified analysis for FUNA-style experiments.")
    parser.add_argument("--model_paths", "-m", nargs="+", required=True)
    parser.add_argument("--method", default=None, help="Method key inside results.json. Default: auto-pick.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--void_raw", type=float, default=1.0)
    parser.add_argument("--void_eps", type=float, default=1e-6)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    scene_meta_cache: Dict[Tuple[str, str], dict] = {}
    records = []
    for model_path in args.model_paths:
        record = load_scene_record(Path(model_path), args.method, scene_meta_cache, args.void_raw, args.void_eps)
        if record is not None:
            records.append(record)

    if not records:
        raise SystemExit("No valid scene records found.")

    depth_edges, depth_labels = make_quantile_bins(records, "depth_p95_m", ["near", "mid", "far"])
    bg_edges, bg_labels = make_quantile_bins(records, "background_ratio", ["low_bg", "mid_bg", "high_bg"])
    tex_edges, tex_labels = make_quantile_bins(records, "texture_strength", ["low_tex", "mid_tex", "high_tex"])

    for record in records:
        record["depth_bin"] = assign_bin(_safe_float(record.get("depth_p95_m", float("nan"))), depth_edges, depth_labels)
        record["background_bin"] = assign_bin(_safe_float(record.get("background_ratio", float("nan"))), bg_edges, bg_labels)
        record["texture_bin"] = assign_bin(_safe_float(record.get("texture_strength", float("nan"))), tex_edges, tex_labels)

    grouped = {
        "water_color": summarize_groups(records, "water_color", args.metrics),
        "degradation_level": summarize_groups(records, "degradation_level", args.metrics),
        "depth_bin": summarize_groups(records, "depth_bin", args.metrics),
        "background_bin": summarize_groups(records, "background_bin", args.metrics),
        "texture_bin": summarize_groups(records, "texture_bin", args.metrics),
    }

    print("group_type\tgroup\t" + "\t".join(["num_scenes"] + args.metrics))
    for group_type, rows in grouped.items():
        for group_name, row in rows.items():
            values = [group_type, group_name, str(row["num_scenes"])]
            for metric in args.metrics:
                value = row.get(metric, float("nan"))
                values.append(f"{value:.6f}" if isinstance(value, float) and math.isfinite(value) else "nan")
            print("\t".join(values))

    if args.output_json:
        payload = {
            "records": records,
            "grouped": grouped,
            "thresholds": {
                "depth_bin_edges": depth_edges,
                "background_bin_edges": bg_edges,
                "texture_bin_edges": tex_edges,
            },
            "metrics": args.metrics,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
