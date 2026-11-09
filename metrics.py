# -*- coding: utf-8 -*-
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
import numpy as np
from argparse import ArgumentParser
from typing import Optional
import math

try:
    import imageio.v3 as iio
except Exception:
    import imageio as iio

import OpenEXR
import Imath


# -------------------- CIEDE2000(ΔE00) 相关 --------------------
def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1.0 + a)) ** 2.4)

def _rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb_lin = _srgb_to_linear(rgb)
    M = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=rgb.dtype,
        device=rgb.device
    )
    xyz = torch.einsum("ij,bjhw->bihw", M, rgb_lin) * 100.0
    return xyz

def _f_lab(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    thr = delta ** 3
    return torch.where(t > thr, t ** (1.0 / 3.0), t / (3.0 * delta * delta) + 4.0 / 29.0)

def _xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    Xn, Yn, Zn = 95.047, 100.0, 108.883
    x = xyz[:, 0:1, :, :] / Xn
    y = xyz[:, 1:2, :, :] / Yn
    z = xyz[:, 2:3, :, :] / Zn
    fx = _f_lab(x)
    fy = _f_lab(y)
    fz = _f_lab(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([L, a, b], dim=1)

def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    return _xyz_to_lab(_rgb_to_xyz(rgb))

def ciede2000_mean(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    lab1 = _rgb_to_lab(img1)
    lab2 = _rgb_to_lab(img2)

    L1, a1, b1 = lab1[:, 0, :, :], lab1[:, 1, :, :], lab1[:, 2, :, :]
    L2, a2, b2 = lab2[:, 0, :, :], lab2[:, 1, :, :], lab2[:, 2, :, :]

    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    Cbar = 0.5 * (C1 + C2)

    pow25_7 = 25.0 ** 7
    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - torch.sqrt(Cbar7 / (Cbar7 + pow25_7 + 1e-12)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = torch.sqrt(a1p * a1p + b1 * b1)
    C2p = torch.sqrt(a2p * a2p + b2 * b2)

    h1p = torch.atan2(b1, a1p) * (180.0 / math.pi)
    h2p = torch.atan2(b2, a2p) * (180.0 / math.pi)
    h1p = torch.remainder(h1p + 360.0, 360.0)
    h2p = torch.remainder(h2p + 360.0, 360.0)

    dLp = L2 - L1
    dCp = C2p - C1p

    hp_diff = h2p - h1p
    hp_diff = torch.where(hp_diff > 180.0, hp_diff - 360.0, hp_diff)
    hp_diff = torch.where(hp_diff < -180.0, hp_diff + 360.0, hp_diff)

    zero_C = (C1p * C2p) < 1e-12
    dhp = torch.where(zero_C, torch.zeros_like(hp_diff), hp_diff)

    dHp = 2.0 * torch.sqrt(torch.clamp(C1p * C2p, min=0.0)) * torch.sin((dhp * (math.pi / 180.0)) / 2.0)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    habs = torch.abs(h1p - h2p)
    hbarp = torch.where(
        zero_C,
        hsum,
        torch.where(
            habs <= 180.0,
            0.5 * hsum,
            torch.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0))
        )
    )

    T = (1.0
         - 0.17 * torch.cos((hbarp - 30.0) * (math.pi / 180.0))
         + 0.24 * torch.cos((2.0 * hbarp) * (math.pi / 180.0))
         + 0.32 * torch.cos((3.0 * hbarp + 6.0) * (math.pi / 180.0))
         - 0.20 * torch.cos((4.0 * hbarp - 63.0) * (math.pi / 180.0)))

    delta_theta = 30.0 * torch.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp ** 7
    Rc = 2.0 * torch.sqrt(Cbarp7 / (Cbarp7 + pow25_7 + 1e-12))

    Sl = 1.0 + (0.015 * (Lbarp - 50.0) ** 2) / torch.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T

    Rt = -torch.sin((2.0 * delta_theta) * (math.pi / 180.0)) * Rc

    kL = kC = kH = 1.0
    dE = torch.sqrt(
        (dLp / (kL * Sl + 1e-12)) ** 2 +
        (dCp / (kC * Sc + 1e-12)) ** 2 +
        (dHp / (kH * Sh + 1e-12)) ** 2 +
        Rt * (dCp / (kC * Sc + 1e-12)) * (dHp / (kH * Sh + 1e-12))
    )
    return dE.mean(dim=(-2, -1))


# -------------------- 读取 RGB --------------------
def readImages(renders_dir, gt_dir):
    renders, gts, image_names = [], [], []
    for fname in os.listdir(renders_dir):
        if not os.path.splitext(fname)[1].lower() in [".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp"]:
            continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


# -------------------- EXR 单通道读取（深度） --------------------
def read_exr_channel(file_path: str, channel_name: str = "R") -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
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
            if channel_name.lower() in lower_map:
                ch = lower_map[channel_name.lower()]
            else:
                cand = [c for c in channels if c.split(".")[-1].lower() == channel_name.lower()]
                if len(cand) == 1:
                    ch = cand[0]
        if ch is None:
            raise KeyError(f"EXR 不含通道 {channel_name}。可用通道: {channels}")
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


# -------------------- 法线读取与解码 --------------------
def _read_normal_exr_rgb(path: str) -> np.ndarray:
    arr = iio.imread(path).astype(np.float32)
    if arr.ndim == 2:
        raise ValueError(f"{path} 不是三通道法线图")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)

def _detect_encoding(arr: np.ndarray) -> str:
    eps = 1e-3
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mn < -0.1 or mx > 1.1:
        return "neg11"
    if mn >= -eps and mx <= 1.0 + eps:
        m = float(np.nanmean(arr))
        return "01" if abs(m - 0.5) <= abs(m - 0.0) else "neg11"
    m = float(np.nanmean(arr))
    if -0.2 <= mn and mx <= 1.2 and 0.2 <= m <= 0.8:
        return "01"
    return "neg11"

def _decode_normals(arr: np.ndarray, enc: str = "auto") -> np.ndarray:
    if enc == "auto":
        enc = _detect_encoding(arr)
    if enc == "01":
        n = arr * 2.0 - 1.0
    elif enc == "neg11":
        n = arr
    else:
        raise ValueError(f"未知编码 {enc}")
    return _normalize(n)

def _valid_mask(n: np.ndarray) -> np.ndarray:
    return np.isfinite(n).all(axis=-1) & (np.linalg.norm(n, axis=-1) > 1e-6)

def _angular_error_deg(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    cos = np.clip(np.sum(n1 * n2, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# -------------------- 深度评估 --------------------
def _align_shapes(a: np.ndarray, b: np.ndarray, mode: str = "strict"):
    if a.shape == b.shape:
        return a, b
    if mode == "strict":
        raise ValueError(f"形状不一致: {a.shape} vs {b.shape}")
    if mode == "crop":
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        def crop(x):
            dh = (x.shape[0] - h) // 2
            dw = (x.shape[1] - w) // 2
            return x[dh:dh+h, dw:dw+w]
        return crop(a), crop(b)
    raise ValueError(f"未知对齐模式 {mode}")

def _compute_void_masks(pred_raw: np.ndarray, gt_raw: np.ndarray, void_depth_raw: float, void_eps: float):
    pred_void = np.abs(pred_raw - float(void_depth_raw)) <= float(void_eps)
    gt_void   = np.abs(gt_raw   - float(void_depth_raw)) <= float(void_eps)
    void_agree = pred_void & gt_void
    void_mismatch = pred_void ^ gt_void
    non_void = (~pred_void) & (~gt_void)
    return pred_void, gt_void, void_agree, void_mismatch, non_void

def _build_base_valid(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float,
    max_depth: float,
    ratio_outlier: float,
    range_relax: float = 0.05,
    eps: float = 1e-8,
    candidate_mask: Optional[np.ndarray] = None,
):
    valid = np.isfinite(pred) & np.isfinite(gt)
    if candidate_mask is not None:
        if candidate_mask.shape != valid.shape:
            raise ValueError(f"candidate_mask 形状不匹配: {candidate_mask.shape} vs {valid.shape}")
        valid &= candidate_mask

    max_relaxed = max_depth * (1.0 + max(range_relax, 0.0))
    valid &= (gt > min_depth) & (gt <= max_relaxed)

    pred_clip = np.clip(pred, min_depth, max_depth)
    valid &= np.isfinite(pred_clip)

    if not np.any(valid):
        return valid, pred_clip

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(
            (pred_clip[valid] + eps) / (gt[valid] + eps),
            (gt[valid] + eps) / (pred_clip[valid] + eps),
        )
    keep = np.zeros_like(valid, bool)
    keep[valid] = ratio <= ratio_outlier
    return valid & keep, pred_clip

def _depth_value_metrics(pred_clip, gt, mask, eps=1e-8):
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    diff = pred_clip[mask] - gt[mask]
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(abs_diff))
    absrel = float(np.mean(abs_diff / (gt[mask] + eps)))
    return rmse, mae, absrel

def _depth_rate_metrics_allpixels(
    pred_clip: np.ndarray,
    gt: np.ndarray,
    non_void_valid_for_ratio: np.ndarray,
    void_agree: Optional[np.ndarray],
    eps: float = 1e-8,
):
    H, W = pred_clip.shape
    total = H * W
    if total == 0:
        return np.nan, np.nan, np.nan

    cond1 = np.zeros((H, W), dtype=bool)
    cond2 = np.zeros((H, W), dtype=bool)
    cond3 = np.zeros((H, W), dtype=bool)

    if void_agree is not None and np.any(void_agree):
        cond1[void_agree] = True
        cond2[void_agree] = True
        cond3[void_agree] = True

    if np.any(non_void_valid_for_ratio):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.maximum(
                (pred_clip[non_void_valid_for_ratio] + eps) / (gt[non_void_valid_for_ratio] + eps),
                (gt[non_void_valid_for_ratio] + eps) / (pred_clip[non_void_valid_for_ratio] + eps),
            )
        cond1[non_void_valid_for_ratio] = ratio < 1.25
        cond2[non_void_valid_for_ratio] = ratio < (1.25 ** 2)
        cond3[non_void_valid_for_ratio] = ratio < (1.25 ** 3)

    return (
        float(cond1.sum() / float(total)),
        float(cond2.sum() / float(total)),
        float(cond3.sum() / float(total)),
    )

def eval_depth_one(
    pred_path,
    gt_path,
    pred_channel="Z",
    gt_channel="R",
    pred_scale=1000.0,   # [0,1] -> 0..1000m
    gt_scale=1000.0,
    min_depth=1e-3,
    max_depth=float("inf"),
    ratio_outlier=10.0,
    range_relax=0.05,
    align="strict",
    void_depth_raw: Optional[float] = 1.0,
    void_eps: float = 1e-6,
) -> dict:
    pred_raw = read_exr_channel(pred_path, pred_channel).astype(np.float32)
    gt_raw   = read_exr_channel(gt_path,   gt_channel).astype(np.float32)
    pred_raw, gt_raw = _align_shapes(pred_raw, gt_raw, mode=align)

    if void_depth_raw is not None:
        pred_void, gt_void, void_agree, void_mismatch, non_void = _compute_void_masks(
            pred_raw, gt_raw, void_depth_raw, void_eps
        )
    else:
        pred_void = gt_void = void_agree = void_mismatch = None
        non_void = None

    pred = pred_raw * float(pred_scale)
    gt   = gt_raw   * float(gt_scale)
    pred, gt = _align_shapes(pred, gt, mode=align)

    pred_clip = np.clip(pred, min_depth, max_depth)

    candidate_nonvoid = non_void if non_void is not None else None
    base_valid_nonvoid, pred_clip = _build_base_valid(
        pred, gt,
        min_depth=min_depth,
        max_depth=max_depth,
        ratio_outlier=ratio_outlier,
        range_relax=range_relax,
        candidate_mask=candidate_nonvoid,
    )
    rmse, mae, absrel = _depth_value_metrics(pred_clip, gt, base_valid_nonvoid)

    max_relaxed = max_depth * (1.0 + max(range_relax, 0.0))
    non_void_valid_for_ratio = np.isfinite(pred_clip) & np.isfinite(gt) & (gt > min_depth) & (gt <= max_relaxed)
    if non_void is not None:
        non_void_valid_for_ratio &= non_void
    if void_mismatch is not None:
        non_void_valid_for_ratio &= (~void_mismatch)

    d1, d2, d3 = _depth_rate_metrics_allpixels(
        pred_clip=pred_clip,
        gt=gt,
        non_void_valid_for_ratio=non_void_valid_for_ratio,
        void_agree=void_agree,
    )

    H, W = pred.shape
    total = float(H * W) if H > 0 and W > 0 else 1.0

    if pred_void is not None:
        void_pred_ratio = float(pred_void.sum() / total)
        void_gt_ratio = float(gt_void.sum() / total)
        void_agree_ratio = float(void_agree.sum() / total)
        void_mismatch_ratio = float(void_mismatch.sum() / total)
        nonvoid_ratio = float(non_void.sum() / total)
    else:
        void_pred_ratio = void_gt_ratio = void_agree_ratio = void_mismatch_ratio = 0.0
        nonvoid_ratio = 1.0

    return {
        "depth.nonvoid_ratio": nonvoid_ratio,
        "depth.void_pred_ratio": void_pred_ratio,
        "depth.void_gt_ratio": void_gt_ratio,
        "depth.void_agree_ratio": void_agree_ratio,
        "depth.void_mismatch_ratio": void_mismatch_ratio,

        "depth.valid_ratio": float(base_valid_nonvoid.mean()) if base_valid_nonvoid.size > 0 else 0.0,
        "depth.value_ratio": float(base_valid_nonvoid.mean()) if base_valid_nonvoid.size > 0 else 0.0,

        "depth.rmse": rmse,
        "depth.mae": mae,
        "depth.absrel": absrel,

        "depth.delta<1.25": d1,
        "depth.delta<1.25^2": d2,
        "depth.delta<1.25^3": d3,
    }


# -------------------- 法线评估（保持原口径） --------------------
def make_normal_keep_mask_from_gt_depth(gt_depth_exr: str, gt_channel: str, eq_value: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    arr = read_exr_channel(gt_depth_exr, gt_channel).astype(np.float32)
    bad = np.abs(arr - float(eq_value)) <= float(eps)
    return ~bad

def eval_normals_one(
    pred_path,
    gt_path,
    gt_depth_path=None,
    pred_enc="auto",
    gt_enc="auto",
    gt_depth_channel="R",
    ignore_depth_eq_value=1.0,
    ignore_depth_eps=1e-6,
) -> dict:
    n_pred = _decode_normals(_read_normal_exr_rgb(pred_path), pred_enc)
    n_gt = _decode_normals(_read_normal_exr_rgb(gt_path), gt_enc)

    valid_norm = _valid_mask(n_pred) & _valid_mask(n_gt)

    denom_mask = np.ones(valid_norm.shape, dtype=bool)
    if gt_depth_path is not None:
        keep_mask = make_normal_keep_mask_from_gt_depth(
            gt_depth_path, gt_depth_channel,
            eq_value=ignore_depth_eq_value, eps=ignore_depth_eps
        )
        if keep_mask.shape[:2] != valid_norm.shape:
            raise ValueError(f"keep_mask 形状不匹配: {keep_mask.shape} vs {valid_norm.shape}")
        denom_mask &= keep_mask

    idx = denom_mask & valid_norm

    if idx.sum() == 0 or denom_mask.sum() == 0:
        return {
            "normal.valid_ratio": 0.0,
            "normal.mean_angle_deg": float("nan"),
            "normal.median_angle_deg": float("nan"),
            "normal.acc@11.25deg": 0.0,
            "normal.acc@22.5deg": 0.0,
            "normal.acc@30deg": 0.0,
        }

    ang = _angular_error_deg(n_pred, n_gt)

    mean_ang = float(np.mean(ang[idx]))
    median_ang = float(np.median(ang[idx]))

    denom = float(np.sum(denom_mask))
    num_11 = np.sum((ang <= 11.25) & valid_norm & denom_mask)
    num_22 = np.sum((ang <= 22.5) & valid_norm & denom_mask)
    num_30 = np.sum((ang <= 30.0) & valid_norm & denom_mask)

    return {
        "normal.valid_ratio": float(idx.mean()),
        "normal.mean_angle_deg": mean_ang,
        "normal.median_angle_deg": median_ang,
        "normal.acc@11.25deg": float(num_11 / denom),
        "normal.acc@22.5deg": float(num_22 / denom),
        "normal.acc@30deg": float(num_30 / denom),
    }


# -------------------- 主评估流程 --------------------
def evaluate(
    model_paths,
    source_path: str = None,
    air_source_path: str = None,
    depth_void_value: float = 1.0,
    depth_void_eps: float = 1e-6,
):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            if source_path is None:
                try:
                    with open(Path(scene_dir) / "scene_paths.json", 'r') as f:
                        sp = json.load(f)
                        source_path_local = sp.get("source_path", None)
                        air_source_path_local = sp.get("air_source_path", None)
                except Exception:
                    source_path_local = None
                    air_source_path_local = None
            else:
                source_path_local = source_path
                air_source_path_local = air_source_path

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                depth_exr_dir = method_dir / "renders_depth_exr"
                normal_exr_dir = method_dir / "renders_normal_exr"

                if source_path_local is None:
                    renders, gts, image_names = readImages(renders_dir, gt_dir)
                else:
                    renders, gts, image_names = readImages(renders_dir, Path(source_path_local) / "images/ColorImage")

                ssims, psnrs, lpipss = [], [], []
                for idx_img in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx_img], gts[idx_img]))
                    psnrs.append(psnr(renders[idx_img], gts[idx_img]))
                    lpipss.append(lpips(renders[idx_img], gts[idx_img], net_type='vgg'))

                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item()
                })

                # Depth
                if os.path.isdir(depth_exr_dir) and (source_path_local is not None):
                    depth_metrics_list = []
                    for name in image_names:
                        stem = os.path.splitext(name)[0]
                        pred_path = depth_exr_dir / f"{stem}.exr"
                        gt_path = Path(source_path_local) / "images" / "DepthImage" / f"{stem}.exr"
                        if pred_path.exists() and gt_path.exists():
                            dm = eval_depth_one(
                                str(pred_path), str(gt_path),
                                pred_channel="Z", gt_channel="R",
                                pred_scale=1000.0, gt_scale=1000.0,
                                void_depth_raw=depth_void_value,
                                void_eps=depth_void_eps,
                            )
                            depth_metrics_list.append(dm)
                    if depth_metrics_list:
                        keys = sorted({k for d in depth_metrics_list for k in d.keys()})
                        depth_mean = {k: float(np.nanmean([d[k] for d in depth_metrics_list])) for k in keys}
                        full_dict[scene_dir][method].update(depth_mean)

                # Normal（保持原逻辑）
                if os.path.isdir(normal_exr_dir) and (source_path_local is not None):
                    normal_metrics_list = []
                    for name in image_names:
                        stem = os.path.splitext(name)[0]
                        pred_path = normal_exr_dir / f"{stem}.exr"
                        gt_path = Path(source_path_local) / "images" / "NormalImage" / f"{stem}.exr"
                        gt_depth_path = Path(source_path_local) / "images" / "DepthImage" / f"{stem}.exr"
                        if pred_path.exists() and gt_path.exists() and gt_depth_path.exists():
                            nm = eval_normals_one(
                                str(pred_path), str(gt_path),
                                gt_depth_path=str(gt_depth_path),
                                pred_enc="auto", gt_enc="auto",
                                gt_depth_channel="R",
                                ignore_depth_eq_value=1.0, ignore_depth_eps=1e-6
                            )
                            normal_metrics_list.append(nm)
                    if normal_metrics_list:
                        keys = sorted({k for d in normal_metrics_list for k in d.keys()})
                        normal_mean = {k: float(np.nanmean([d[k] for d in normal_metrics_list])) for k in keys}
                        full_dict[scene_dir][method].update(normal_mean)

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True, ensure_ascii=False)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, "reason:", e)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="UW_PGSR evaluation: RGB/Depth/Normal")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--source_path', '-s', required=False, type=str, default=None)
    parser.add_argument('--air_source_path', '-sa', required=False, type=str, default=None)

    # 默认就按你的数据：raw 值域 [0,1]，void=1.0
    parser.add_argument('--depth_void_value', type=float, default=1.0,
                        help='void 哨兵值（raw 深度 [0,1] 下），默认 1.0 表示 1000m')
    parser.add_argument('--depth_void_eps', type=float, default=1e-6,
                        help='void 判断容差（raw 值域下），默认 1e-6')

    args = parser.parse_args()

    # 兜底：如果用户误传了 1000 这种明显不在 [0,1] 的值，自动改成 1.0
    if args.depth_void_value is not None and args.depth_void_value > 2.0:
        print(f"[Info] depth_void_value={args.depth_void_value} 看起来不是 raw([0,1]) 值，自动改为 1.0")
        args.depth_void_value = 1.0

    evaluate(
        args.model_paths,
        args.source_path,
        args.air_source_path,
        depth_void_value=args.depth_void_value,
        depth_void_eps=args.depth_void_eps,
    )
