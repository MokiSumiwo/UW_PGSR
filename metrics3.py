# -*- coding: utf-8 -*-
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

from pathlib import Path
import os
from PIL import Image
import torch
import torch.nn.functional as F
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

# -------------------- pyiqa: NIQE --------------------
try:
    import pyiqa
    _HAS_PYIQA = True
except Exception:
    pyiqa = None
    _HAS_PYIQA = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NIQE_METRIC = None


# -------------------- CIE / Lab 相关 --------------------
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


# -------------------- 通用 RGB 读取 --------------------
def _ensure_rgb_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported channel shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image ndim: {arr.ndim}")

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    mn = float(arr.min()) if arr.size > 0 else 0.0
    mx = float(arr.max()) if arr.size > 0 else 1.0

    if mx > 1.0 or mn < 0.0:
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

    return np.clip(arr, 0.0, 1.0)


def _load_rgb_tensor(img_path: Path) -> torch.Tensor:
    try:
        img = Image.open(img_path).convert("RGB")
        return tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].to(DEVICE)
    except Exception:
        arr = iio.imread(str(img_path))
        arr = _ensure_rgb_hwc(arr)
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        return ten


def readPredImages(renders_dir: Path):
    renders, image_names = [], []
    if renders_dir is None or (not renders_dir.exists()):
        return renders, image_names

    for fname in sorted(os.listdir(renders_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp"]:
            continue
        pred_p = renders_dir / fname
        try:
            render = _load_rgb_tensor(pred_p)
            renders.append(render)
            image_names.append(fname)
        except Exception as e:
            print(f"[readPredImages] skip {fname}: {e}")
            continue
    return renders, image_names


def readMatchedGTs(gt_dir: Optional[Path], image_names):
    gt_map = {}
    if gt_dir is None or (not gt_dir.exists()):
        return gt_map
    for fname in image_names:
        gt_p = gt_dir / fname
        if not gt_p.exists():
            continue
        try:
            gt_map[fname] = _load_rgb_tensor(gt_p)
        except Exception as e:
            print(f"[readMatchedGTs] skip {fname}: {e}")
            continue
    return gt_map


# -------------------- torch 工具：无参考指标全程 GPU --------------------
def _resize_tensor_for_nr(img: torch.Tensor, max_side: int = 640) -> torch.Tensor:
    if max_side is None or max_side <= 0:
        return img
    h, w = img.shape[-2], img.shape[-1]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return F.interpolate(img, size=(nh, nw), mode="area")


def _torch_quantile_1d(x: torch.Tensor, q: float) -> torch.Tensor:
    x = x.reshape(-1)
    if x.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)

    if hasattr(torch, "quantile"):
        qv = torch.tensor(q, device=x.device, dtype=x.dtype)
        return torch.quantile(x, qv)

    xs, _ = torch.sort(x)
    pos = q * (xs.numel() - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _alpha_trimmed_torch(arr: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    x = arr.reshape(-1)
    if x.numel() == 0:
        return x
    xs, _ = torch.sort(x)
    n = xs.numel()
    k = int(math.floor(alpha * n))
    if 2 * k >= n:
        return xs
    return xs[k:n - k]


def _sobel_mag_torch(channel: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=channel.dtype, device=channel.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=channel.dtype, device=channel.device).view(1, 1, 3, 3)
    x = F.pad(channel, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    return torch.sqrt(torch.clamp(gx * gx + gy * gy, min=0.0))


def _eme_torch(x: torch.Tensor, block_size: int = 8, eps: float = 1e-8) -> torch.Tensor:
    B, C, H, W = x.shape
    assert C == 1

    if H < block_size or W < block_size:
        x_max = x.amax(dim=(-2, -1)).squeeze(-1).squeeze(-1)
        x_min = x.amin(dim=(-2, -1)).squeeze(-1).squeeze(-1)
        bad = (x_min < eps) | (x_max < eps)
        val = 20.0 * torch.log((x_max + eps) / (x_min + eps))
        return torch.where(bad, torch.zeros_like(val), val)

    h = (H // block_size) * block_size
    w = (W // block_size) * block_size
    if h == 0 or w == 0:
        return torch.zeros(B, dtype=x.dtype, device=x.device)

    x = x[:, :, :h, :w]
    x = x.view(B, 1, h // block_size, block_size, w // block_size, block_size)
    bmax = x.amax(dim=(3, 5))
    bmin = x.amin(dim=(3, 5))

    valid = (bmax > eps) & (bmin > eps)
    vals = torch.where(valid, torch.log((bmax + eps) / (bmin + eps)), torch.zeros_like(bmax))
    count = vals.shape[-1] * vals.shape[-2]
    return (2.0 / float(count)) * vals.sum(dim=(-2, -1)).squeeze(1)


def _uicm_torch(img: torch.Tensor) -> torch.Tensor:
    B = img.shape[0]
    out = []
    for i in range(B):
        R = img[i, 0]
        G = img[i, 1]
        Bc = img[i, 2]

        rg = R - G
        yb = 0.5 * (R + G) - Bc

        rg_t = _alpha_trimmed_torch(rg, alpha=0.1)
        yb_t = _alpha_trimmed_torch(yb, alpha=0.1)

        if rg_t.numel() == 0 or yb_t.numel() == 0:
            out.append(torch.tensor(float("nan"), device=img.device, dtype=img.dtype))
            continue

        mu_rg = rg_t.mean()
        mu_yb = yb_t.mean()
        sigma_rg = rg_t.var(unbiased=False)
        sigma_yb = yb_t.var(unbiased=False)

        chroma_bias = torch.sqrt(mu_rg * mu_rg + mu_yb * mu_yb)
        chroma_var = torch.sqrt(sigma_rg + sigma_yb)
        out.append((-0.0268 * chroma_bias) + (0.1586 * chroma_var))

    return torch.stack(out)


def _uism_torch(img: torch.Tensor) -> torch.Tensor:
    R = img[:, 0:1]
    G = img[:, 1:2]
    Bc = img[:, 2:3]

    Rs = _sobel_mag_torch(R) * R
    Gs = _sobel_mag_torch(G) * G
    Bs = _sobel_mag_torch(Bc) * Bc

    r_eme = _eme_torch(Rs)
    g_eme = _eme_torch(Gs)
    b_eme = _eme_torch(Bs)

    return 0.299 * r_eme + 0.587 * g_eme + 0.114 * b_eme


def _uiconm_torch(img: torch.Tensor, block_size: int = 8, eps: float = 1e-8) -> torch.Tensor:
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    B, C, H, W = gray.shape
    assert C == 1

    if H < block_size or W < block_size:
        xmax = gray.amax(dim=(-2, -1)).squeeze(-1).squeeze(-1)
        xmin = gray.amin(dim=(-2, -1)).squeeze(-1).squeeze(-1)
        return (xmax - xmin) / (xmax + xmin + eps)

    h = (H // block_size) * block_size
    w = (W // block_size) * block_size
    if h == 0 or w == 0:
        return torch.zeros(B, dtype=img.dtype, device=img.device)

    gray = gray[:, :, :h, :w]
    gray = gray.view(B, 1, h // block_size, block_size, w // block_size, block_size)
    bmax = gray.amax(dim=(3, 5))
    bmin = gray.amin(dim=(3, 5))
    val = (bmax - bmin) / (bmax + bmin + eps)
    return val.mean(dim=(-2, -1)).squeeze(1)


def compute_uiqm_torch(img: torch.Tensor) -> torch.Tensor:
    return 0.0282 * _uicm_torch(img) + 0.2953 * _uism_torch(img) + 3.5753 * _uiconm_torch(img)


def compute_uciqe_torch(img: torch.Tensor) -> torch.Tensor:
    lab = _rgb_to_lab(img)
    L = lab[:, 0:1] / 100.0
    a = lab[:, 1:2]
    b = lab[:, 2:3]

    chroma = torch.sqrt(a * a + b * b)
    sigma_c = chroma.flatten(1).std(dim=1, unbiased=False)

    con_l_list = []
    for i in range(img.shape[0]):
        q99 = _torch_quantile_1d(L[i], 0.99)
        q01 = _torch_quantile_1d(L[i], 0.01)
        con_l_list.append(q99 - q01)
    con_l = torch.stack(con_l_list)

    sat = chroma / torch.sqrt(chroma * chroma + L * L + 1e-12)
    mu_s = sat.flatten(1).mean(dim=1)

    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


def _get_niqe_metric():
    global _NIQE_METRIC
    if not _HAS_PYIQA:
        return None
    if _NIQE_METRIC is None:
        device_str = "cuda:0" if DEVICE.type == "cuda" else "cpu"
        _NIQE_METRIC = pyiqa.create_metric("niqe", device=device_str)
        _NIQE_METRIC.eval()
    return _NIQE_METRIC


def compute_niqe_torch(img: torch.Tensor) -> float:
    metric = _get_niqe_metric()
    if metric is None:
        return float("nan")
    try:
        with torch.no_grad():
            score = metric(img)
        return float(score.reshape(-1)[0].item())
    except Exception:
        return float("nan")


def eval_no_reference_metrics_from_tensor(pred_rgb: torch.Tensor, max_side: int = 640) -> dict:
    img = pred_rgb.detach().float().clamp(0.0, 1.0)
    img = _resize_tensor_for_nr(img, max_side=max_side)

    with torch.no_grad():
        uiqm = compute_uiqm_torch(img).reshape(-1)[0].item()
        uciqe = compute_uciqe_torch(img).reshape(-1)[0].item()
        niqe = compute_niqe_torch(img)

    return {
        "UIQM": float(uiqm),
        "UCIQE": float(uciqe),
        "NIQE": float(niqe),
    }


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


# -------------------- 深度评估工具 --------------------
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
            return x[dh:dh + h, dw:dw + w]

        return crop(a), crop(b)
    raise ValueError(f"未知对齐模式 {mode}")


def _align_shapes_3(a: np.ndarray, b: np.ndarray, m: Optional[np.ndarray], mode: str = "strict"):
    a2, b2 = _align_shapes(a, b, mode=mode)
    if m is None:
        return a2, b2, None
    if m.shape == a2.shape:
        return a2, b2, m
    if mode == "strict":
        raise ValueError(f"mask 形状不一致: {m.shape} vs {a2.shape}")
    if mode == "crop":
        h = min(a2.shape[0], m.shape[0])
        w = min(a2.shape[1], m.shape[1])

        def crop2(x):
            dh = (x.shape[0] - h) // 2
            dw = (x.shape[1] - w) // 2
            return x[dh:dh + h, dw:dw + w]

        return crop2(a2), crop2(b2), crop2(m)
    raise ValueError(f"未知对齐模式 {mode}")


def _compute_void_masks(pred_raw: np.ndarray, gt_raw: np.ndarray, void_depth_raw: float, void_eps: float):
    pred_void = np.abs(pred_raw - float(void_depth_raw)) <= float(void_eps)
    gt_void = np.abs(gt_raw - float(void_depth_raw)) <= float(void_eps)
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


def _depth_rate_metrics_masked(
    pred_clip: np.ndarray,
    gt: np.ndarray,
    non_void_valid_for_ratio: np.ndarray,
    eval_mask: np.ndarray,
    void_agree: Optional[np.ndarray] = None,
    eps: float = 1e-8,
):
    total = float(np.sum(eval_mask))
    if total <= 0.0:
        return np.nan, np.nan, np.nan

    cond1 = np.zeros_like(eval_mask, dtype=bool)
    cond2 = np.zeros_like(eval_mask, dtype=bool)
    cond3 = np.zeros_like(eval_mask, dtype=bool)

    if void_agree is not None and np.any(void_agree):
        va = void_agree & eval_mask
        cond1[va] = True
        cond2[va] = True
        cond3[va] = True

    idx = non_void_valid_for_ratio & eval_mask
    if np.any(idx):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.maximum(
                (pred_clip[idx] + eps) / (gt[idx] + eps),
                (gt[idx] + eps) / (pred_clip[idx] + eps),
            )
        cond1[idx] = ratio < 1.25
        cond2[idx] = ratio < (1.25 ** 2)
        cond3[idx] = ratio < (1.25 ** 3)

    return (
        float(cond1.sum() / total),
        float(cond2.sum() / total),
        float(cond3.sum() / total),
    )


def eval_depth_one(
    pred_path,
    gt_path,
    pred_channel="Z",
    gt_channel="R",
    pred_scale=1000.0,
    gt_scale=1000.0,
    min_depth=1e-3,
    max_depth=float("inf"),
    ratio_outlier=10.0,
    range_relax=0.05,
    align="strict",
    void_depth_raw: Optional[float] = 1.0,
    void_eps: float = 1e-6,
    eval_mask: Optional[np.ndarray] = None,
) -> dict:
    pred_raw = read_exr_channel(pred_path, pred_channel).astype(np.float32)
    gt_raw = read_exr_channel(gt_path, gt_channel).astype(np.float32)

    if eval_mask is not None:
        pred_raw, gt_raw, eval_mask = _align_shapes_3(pred_raw, gt_raw, eval_mask.astype(bool), mode="crop")
    else:
        pred_raw, gt_raw = _align_shapes(pred_raw, gt_raw, mode=align)

    if void_depth_raw is not None:
        pred_void, gt_void, void_agree, void_mismatch, non_void = _compute_void_masks(
            pred_raw, gt_raw, void_depth_raw, void_eps
        )
    else:
        pred_void = gt_void = void_agree = void_mismatch = None
        non_void = None

    pred = pred_raw * float(pred_scale)
    gt = gt_raw * float(gt_scale)

    if eval_mask is not None:
        pred, gt, eval_mask = _align_shapes_3(pred, gt, eval_mask.astype(bool), mode="crop")
    else:
        pred, gt = _align_shapes(pred, gt, mode=align)

    pred_clip = np.clip(pred, min_depth, max_depth)

    candidate_nonvoid = non_void if non_void is not None else None
    if eval_mask is not None:
        candidate_nonvoid = eval_mask if candidate_nonvoid is None else (candidate_nonvoid & eval_mask)

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

    if eval_mask is None:
        H, W = pred_clip.shape
        total = float(H * W) if H > 0 and W > 0 else 1.0

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
                    (pred_clip[non_void_valid_for_ratio] + 1e-8) / (gt[non_void_valid_for_ratio] + 1e-8),
                    (gt[non_void_valid_for_ratio] + 1e-8) / (pred_clip[non_void_valid_for_ratio] + 1e-8),
                )
            cond1[non_void_valid_for_ratio] = ratio < 1.25
            cond2[non_void_valid_for_ratio] = ratio < (1.25 ** 2)
            cond3[non_void_valid_for_ratio] = ratio < (1.25 ** 3)

        d1 = float(cond1.sum() / total)
        d2 = float(cond2.sum() / total)
        d3 = float(cond3.sum() / total)

        if pred_void is not None:
            void_pred_ratio = float(pred_void.sum() / total)
            void_gt_ratio = float(gt_void.sum() / total)
            void_agree_ratio = float(void_agree.sum() / total)
            void_mismatch_ratio = float(void_mismatch.sum() / total)
            nonvoid_ratio = float(non_void.sum() / total)
        else:
            void_pred_ratio = void_gt_ratio = void_agree_ratio = void_mismatch_ratio = 0.0
            nonvoid_ratio = 1.0

        valid_ratio = float(base_valid_nonvoid.mean()) if base_valid_nonvoid.size > 0 else 0.0
    else:
        d1, d2, d3 = _depth_rate_metrics_masked(
            pred_clip=pred_clip,
            gt=gt,
            non_void_valid_for_ratio=non_void_valid_for_ratio,
            eval_mask=eval_mask,
            void_agree=void_agree,
        )

        total = float(np.sum(eval_mask)) if np.sum(eval_mask) > 0 else 1.0
        if pred_void is not None:
            void_pred_ratio = float(np.sum(pred_void & eval_mask) / total)
            void_gt_ratio = float(np.sum(gt_void & eval_mask) / total)
            void_agree_ratio = float(np.sum(void_agree & eval_mask) / total)
            void_mismatch_ratio = float(np.sum(void_mismatch & eval_mask) / total)
            nonvoid_ratio = float(np.sum((non_void & eval_mask)) / total)
        else:
            void_pred_ratio = void_gt_ratio = void_agree_ratio = void_mismatch_ratio = 0.0
            nonvoid_ratio = 1.0

        valid_ratio = float(np.sum(base_valid_nonvoid) / total)

    return {
        "depth.nonvoid_ratio": nonvoid_ratio,
        "depth.void_pred_ratio": void_pred_ratio,
        "depth.void_gt_ratio": void_gt_ratio,
        "depth.void_agree_ratio": void_agree_ratio,
        "depth.void_mismatch_ratio": void_mismatch_ratio,
        "depth.valid_ratio": valid_ratio,
        "depth.value_ratio": valid_ratio,
        "depth.rmse": rmse,
        "depth.mae": mae,
        "depth.absrel": absrel,
        "depth.delta<1.25": d1,
        "depth.delta<1.25^2": d2,
        "depth.delta<1.25^3": d3,
    }


# -------------------- 法线评估 --------------------
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
    eval_mask: Optional[np.ndarray] = None,
) -> dict:
    n_pred = _decode_normals(_read_normal_exr_rgb(pred_path), pred_enc)
    n_gt = _decode_normals(_read_normal_exr_rgb(gt_path), gt_enc)

    valid_norm = _valid_mask(n_pred) & _valid_mask(n_gt)
    denom_mask = np.ones(valid_norm.shape, dtype=bool)

    def _center_crop_2d(x, h, w):
        dh = (x.shape[0] - h) // 2
        dw = (x.shape[1] - w) // 2
        return x[dh:dh + h, dw:dw + w]

    if gt_depth_path is not None:
        keep_mask = make_normal_keep_mask_from_gt_depth(
            gt_depth_path, gt_depth_channel,
            eq_value=ignore_depth_eq_value, eps=ignore_depth_eps
        )
        if keep_mask.shape[:2] != valid_norm.shape:
            h = min(keep_mask.shape[0], valid_norm.shape[0])
            w = min(keep_mask.shape[1], valid_norm.shape[1])
            keep_mask = _center_crop_2d(keep_mask, h, w)
            valid_norm = _center_crop_2d(valid_norm, h, w)
            n_pred = n_pred[(n_pred.shape[0] - h) // 2:(n_pred.shape[0] - h) // 2 + h,
                            (n_pred.shape[1] - w) // 2:(n_pred.shape[1] - w) // 2 + w, :]
            n_gt = n_gt[(n_gt.shape[0] - h) // 2:(n_gt.shape[0] - h) // 2 + h,
                        (n_gt.shape[1] - w) // 2:(n_gt.shape[1] - w) // 2 + w, :]
        denom_mask &= keep_mask

    if eval_mask is not None:
        if eval_mask.shape != denom_mask.shape:
            h = min(eval_mask.shape[0], denom_mask.shape[0])
            w = min(eval_mask.shape[1], denom_mask.shape[1])
            eval_mask = _center_crop_2d(eval_mask.astype(bool), h, w)
            denom_mask = _center_crop_2d(denom_mask, h, w)
            valid_norm = _center_crop_2d(valid_norm, h, w)
            n_pred = n_pred[(n_pred.shape[0] - h) // 2:(n_pred.shape[0] - h) // 2 + h,
                            (n_pred.shape[1] - w) // 2:(n_pred.shape[1] - w) // 2 + w, :]
            n_gt = n_gt[(n_gt.shape[0] - h) // 2:(n_gt.shape[0] - h) // 2 + h,
                        (n_gt.shape[1] - w) // 2:(n_gt.shape[1] - w) // 2 + w, :]
        denom_mask &= eval_mask.astype(bool)

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
        "normal.valid_ratio": float(np.sum(idx) / denom),
        "normal.mean_angle_deg": mean_ang,
        "normal.median_angle_deg": median_ang,
        "normal.acc@11.25deg": float(num_11 / denom),
        "normal.acc@22.5deg": float(num_22 / denom),
        "normal.acc@30deg": float(num_30 / denom),
    }


# -------------------- 前景 mask（由渲染深度决定） --------------------
def _bbox_from_mask(mask: np.ndarray, pad: int = 0):
    H, W = mask.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, H, 0, W
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if pad > 0:
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(H, y1 + pad)
        x1 = min(W, x1 + pad)
    if y1 <= y0 or x1 <= x0:
        return 0, H, 0, W
    return y0, y1, x0, x1


def build_foreground_mask_from_render_depth(
    depth_exr_path: str,
    channel: str = "Z",
    depth_scale: float = 1000.0,
    void_raw: float = 1.0,
    void_eps: float = 1e-6,
    band_ratio: float = 0.8,
    fg_ratio_in_band: float = 0.5,
    hist_bins: int = 256,
    min_valid_px: int = 1024,
) -> np.ndarray:
    raw = read_exr_channel(depth_exr_path, channel).astype(np.float32)
    valid = np.isfinite(raw) & (np.abs(raw - float(void_raw)) > float(void_eps))

    if int(valid.sum()) < int(min_valid_px):
        return valid

    depth_m = raw * float(depth_scale)
    vals = depth_m[valid]
    if vals.size == 0:
        return valid

    vmin, vmax = float(np.percentile(vals, 1.0)), float(np.percentile(vals, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return valid

    hist, edges = np.histogram(vals, bins=int(hist_bins), range=(vmin, vmax))
    midx = int(np.argmax(hist))
    d_mode = 0.5 * (edges[midx] + edges[midx + 1])

    dist = np.abs(depth_m - d_mode)
    dist_valid = dist[valid]
    k = int(np.ceil(float(band_ratio) * dist_valid.size))
    k = max(1, min(k, dist_valid.size))
    thr = float(np.partition(dist_valid, k - 1)[k - 1])
    band = valid & (dist <= thr)

    if int(band.sum()) < 16:
        return valid

    q = float(np.clip(fg_ratio_in_band, 0.0, 1.0))
    cutoff = float(np.quantile(depth_m[band], q))
    fg = band & (depth_m <= cutoff)
    if int(fg.sum()) < 16:
        return band
    return fg


def masked_psnr_torch(pred: torch.Tensor, gt: torch.Tensor, mask01: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = mask01
    denom = torch.clamp(m.sum() * pred.shape[1], min=1.0)
    diff = (pred - gt) * m
    mse = (diff * diff).sum() / denom
    return 10.0 * torch.log10(torch.clamp(1.0 / torch.clamp(mse, min=eps), min=1.0))


def _resolve_color_dir(base_path: Optional[str]) -> Optional[Path]:
    if base_path is None:
        return None
    if not str(base_path).strip():
        return None

    p = Path(base_path)
    if (p / "images" / "ColorImage").is_dir():
        return p / "images" / "ColorImage"
    if p.is_dir():
        return p
    return None


def _infer_air_source_path_from_source(source_path: Optional[str]) -> Optional[Path]:
    """
    根据带水体 source_path 自动推断去水体 GT 路径。

    规则示例：
        source_path = /mnt/d/home/ycy/V3_Dive3/with/blue/4
    则推断：
        /mnt/d/home/ycy/V3_Dive3/with/air

    即取 source_path 的前两级目录，再拼接 air。
    若不存在 air 目录，则返回 None。
    """
    if source_path is None:
        return None
    if not str(source_path).strip():
        return None

    p = Path(source_path)
    try:
        air_dir = p.parent.parent / "air"
    except Exception:
        return None

    if air_dir.is_dir():
        return air_dir
    return None


def _try_find_air_renders_dir(method_dir: Path) -> Optional[Path]:
    cands = ["renders_air", "render_air", "rendersAir", "renderAir"]
    for c in cands:
        d = method_dir / c
        if d.is_dir():
            return d
    return None


def _try_find_air_bgblack_renders_dir(method_dir: Path) -> Optional[Path]:
    cands = [
        "renders_air_bgblack",
        "render_air_bgblack",
        "rendersAirBgblack",
        "renderAirBgblack",
        "renders_air_bg_black",
        "render_air_bg_black",
    ]
    for c in cands:
        d = method_dir / c
        if d.is_dir():
            return d
    return None


def _metric_list_mean(vals):
    if len(vals) == 0:
        return float("nan")
    if torch.is_tensor(vals[0]):
        vv = [x.detach().float().reshape(-1).mean() for x in vals]
        return torch.stack(vv).mean().item()
    return float(np.nanmean(vals))


def _eval_rgb_set(
    renders_dir: Path,
    gt_dir: Optional[Path],
    fg_masks_by_stem: dict,
    args,
    prefix: str = "",
    compute_no_ref: bool = False,
):
    renders, image_names = readPredImages(renders_dir)
    gt_map = readMatchedGTs(gt_dir, image_names)

    print(f"[{prefix or 'RGB'}] pred={len(renders)}, matched_gt={len(gt_map)}, dir={renders_dir}")

    ssims, psnrs, lpipss = [], [], []
    fg_ssims, fg_psnrs, fg_lpipss = [], [], []
    fg_coverages = []

    nr_uiqms, nr_uciqes, nr_niqes = [], [], []
    fg_nr_uiqms, fg_nr_uciqes, fg_nr_niqes = [], [], []

    if len(renders) == 0:
        out = {}

        if not args.foreground_only:
            out.update({
                f"{prefix}SSIM": float("nan"),
                f"{prefix}PSNR": float("nan"),
                f"{prefix}LPIPS": float("nan"),
            })

        if compute_no_ref:
            out.update({
                f"{prefix}UIQM": float("nan"),
                f"{prefix}UCIQE": float("nan"),
                f"{prefix}NIQE": float("nan"),
            })

        if args.eval_foreground:
            out.update({
                f"{prefix}FG_SSIM": float("nan"),
                f"{prefix}FG_PSNR": float("nan"),
                f"{prefix}FG_LPIPS": float("nan"),
                f"{prefix}FG_Coverage": 0.0,
            })
            if compute_no_ref:
                out.update({
                    f"{prefix}FG_UIQM": float("nan"),
                    f"{prefix}FG_UCIQE": float("nan"),
                    f"{prefix}FG_NIQE": float("nan"),
                })
        return out, []

    with torch.no_grad():
        for idx_img in tqdm(range(len(renders)), desc=f"{prefix}Metric evaluation progress"):
            pred_rgb = renders[idx_img]
            name = image_names[idx_img]
            stem = os.path.splitext(name)[0]
            gt_rgb = gt_map.get(name, None)

            if compute_no_ref:
                nr = eval_no_reference_metrics_from_tensor(pred_rgb, max_side=args.nr_max_side)
                nr_uiqms.append(nr["UIQM"])
                nr_uciqes.append(nr["UCIQE"])
                nr_niqes.append(nr["NIQE"])

            if (gt_rgb is not None) and (not args.foreground_only):
                ssims.append(ssim(pred_rgb, gt_rgb))
                psnrs.append(psnr(pred_rgb, gt_rgb))
                lpipss.append(lpips(pred_rgb, gt_rgb, net_type="vgg"))

            if args.eval_foreground and (stem in fg_masks_by_stem):
                fg_mask_np = fg_masks_by_stem[stem]
                Hm, Wm = fg_mask_np.shape
                Hr, Wr = pred_rgb.shape[-2], pred_rgb.shape[-1]

                if (Hm != Hr) or (Wm != Wr):
                    h = min(Hm, Hr)
                    w = min(Wm, Wr)

                    def _center_crop_np(x):
                        dh = (x.shape[0] - h) // 2
                        dw = (x.shape[1] - w) // 2
                        return x[dh:dh + h, dw:dw + w]

                    def _center_crop_torch(x):
                        dh = (x.shape[-2] - h) // 2
                        dw = (x.shape[-1] - w) // 2
                        return x[..., dh:dh + h, dw:dw + w]

                    fg_mask_np = _center_crop_np(fg_mask_np)
                    pred_rgb_c = _center_crop_torch(pred_rgb)
                    gt_rgb_c = _center_crop_torch(gt_rgb) if gt_rgb is not None else None
                else:
                    pred_rgb_c = pred_rgb
                    gt_rgb_c = gt_rgb

                mask01 = torch.from_numpy(fg_mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                fg_coverages.append(float(mask01.mean().item()))

                y0, y1, x0, x1 = _bbox_from_mask(fg_mask_np, pad=int(args.fg_bbox_pad))
                pred_crop = pred_rgb_c[..., y0:y1, x0:x1]
                mask_crop = mask01[..., y0:y1, x0:x1]

                if compute_no_ref:
                    nr_fg = eval_no_reference_metrics_from_tensor(pred_crop, max_side=args.nr_max_side)
                    fg_nr_uiqms.append(nr_fg["UIQM"])
                    fg_nr_uciqes.append(nr_fg["UCIQE"])
                    fg_nr_niqes.append(nr_fg["NIQE"])

                if gt_rgb_c is not None:
                    gt_crop = gt_rgb_c[..., y0:y1, x0:x1]
                    pred_fill = pred_crop * mask_crop + gt_crop * (1.0 - mask_crop)

                    fg_ssims.append(ssim(pred_fill, gt_crop))
                    fg_psnrs.append(masked_psnr_torch(pred_crop, gt_crop, mask_crop))
                    fg_lpipss.append(lpips(pred_fill, gt_crop, net_type="vgg"))

    out = {}

    if not args.foreground_only:
        out.update({
            f"{prefix}SSIM": _metric_list_mean(ssims),
            f"{prefix}PSNR": _metric_list_mean(psnrs),
            f"{prefix}LPIPS": _metric_list_mean(lpipss),
        })

    if compute_no_ref:
        out.update({
            f"{prefix}UIQM": float(np.nanmean(nr_uiqms)) if nr_uiqms else float("nan"),
            f"{prefix}UCIQE": float(np.nanmean(nr_uciqes)) if nr_uciqes else float("nan"),
            f"{prefix}NIQE": float(np.nanmean(nr_niqes)) if nr_niqes else float("nan"),
        })

    if args.eval_foreground:
        out.update({
            f"{prefix}FG_SSIM": _metric_list_mean(fg_ssims),
            f"{prefix}FG_PSNR": _metric_list_mean(fg_psnrs),
            f"{prefix}FG_LPIPS": _metric_list_mean(fg_lpipss),
            f"{prefix}FG_Coverage": float(np.mean(fg_coverages)) if fg_coverages else 0.0,
        })
        if compute_no_ref:
            out.update({
                f"{prefix}FG_UIQM": float(np.nanmean(fg_nr_uiqms)) if fg_nr_uiqms else float("nan"),
                f"{prefix}FG_UCIQE": float(np.nanmean(fg_nr_uciqes)) if fg_nr_uciqes else float("nan"),
                f"{prefix}FG_NIQE": float(np.nanmean(fg_nr_niqes)) if fg_nr_niqes else float("nan"),
            })

    return out, image_names


# -------------------- 主评估流程 --------------------
def evaluate(
    model_paths,
    source_path: str = None,
    air_source_path: str = None,
    depth_void_value: float = 1.0,
    depth_void_eps: float = 1e-6,
    args=None,
):
    full_dict = {}
    print("")

    class _A:
        pass

    if args is None:
        args = _A()
        args.eval_foreground = False
        args.foreground_only = False
        args.fg_band_ratio = 0.8
        args.fg_ratio_in_band = 0.5
        args.fg_hist_bins = 256
        args.fg_min_valid_px = 1024
        args.fg_bbox_pad = 8
        args.nr_max_side = 640

    if not _HAS_PYIQA:
        print("[Warning] pyiqa 导入失败，NIQE 将输出 NaN。请安装: pip install pyiqa")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            if source_path is None:
                try:
                    with open(Path(scene_dir) / "scene_paths.json", "r") as f:
                        sp = json.load(f)
                        source_path_local = sp.get("source_path", None)
                        air_source_path_local = sp.get("air_source_path", None)

                        if air_source_path_local is None or not str(air_source_path_local).strip():
                            inferred_air = _infer_air_source_path_from_source(source_path_local)
                            if inferred_air is not None:
                                air_source_path_local = str(inferred_air)
                                print(f"[Info] auto inferred air_source_path: {air_source_path_local}")
                            else:
                                air_source_path_local = None
                                print(f"[Info] no air GT found for source_path={source_path_local}")
                except Exception:
                    source_path_local = None
                    air_source_path_local = None
            else:
                source_path_local = source_path
                air_source_path_local = air_source_path

                if air_source_path_local is None or not str(air_source_path_local).strip():
                    inferred_air = _infer_air_source_path_from_source(source_path_local)
                    if inferred_air is not None:
                        air_source_path_local = str(inferred_air)
                        print(f"[Info] auto inferred air_source_path: {air_source_path_local}")
                    else:
                        air_source_path_local = None
                        print(f"[Info] no air GT found for source_path={source_path_local}")

            gt_color_dir = _resolve_color_dir(source_path_local)
            gt_air_color_dir = _resolve_color_dir(air_source_path_local)

            for method in sorted(os.listdir(test_dir)):
                print("Method:", method)
                full_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir_default = method_dir / "gt"
                renders_dir = method_dir / "renders"
                depth_exr_dir = method_dir / "renders_depth_exr"
                normal_exr_dir = method_dir / "renders_normal_exr"

                renders_air_dir = _try_find_air_renders_dir(method_dir)
                renders_air_bgblack_dir = _try_find_air_bgblack_renders_dir(method_dir)

                render_gt_dir = gt_dir_default if gt_color_dir is None else gt_color_dir

                if gt_air_color_dir is not None and Path(gt_air_color_dir).exists():
                    render_air_gt_dir = gt_air_color_dir
                else:
                    gt_air_default = method_dir / "gt_air"
                    render_air_gt_dir = gt_air_default if gt_air_default.is_dir() else None

                fg_masks_by_stem = {}
                if args.eval_foreground and depth_exr_dir.is_dir():
                    for fname in sorted(os.listdir(renders_dir)) if renders_dir.is_dir() else []:
                        stem = os.path.splitext(fname)[0]
                        pred_depth_path = depth_exr_dir / f"{stem}.exr"
                        if pred_depth_path.exists():
                            try:
                                fg_masks_by_stem[stem] = build_foreground_mask_from_render_depth(
                                    str(pred_depth_path),
                                    channel="Z",
                                    depth_scale=1000.0,
                                    void_raw=depth_void_value,
                                    void_eps=depth_void_eps,
                                    band_ratio=args.fg_band_ratio,
                                    fg_ratio_in_band=args.fg_ratio_in_band,
                                    hist_bins=args.fg_hist_bins,
                                    min_valid_px=args.fg_min_valid_px,
                                )
                            except Exception:
                                pass

                rgb_metrics, image_names = _eval_rgb_set(
                    renders_dir=renders_dir,
                    gt_dir=render_gt_dir,
                    fg_masks_by_stem=fg_masks_by_stem,
                    args=args,
                    prefix="",
                    compute_no_ref=False,
                )
                full_dict[scene_dir][method].update(rgb_metrics)

                if renders_air_dir is not None:
                    air_metrics, _ = _eval_rgb_set(
                        renders_dir=renders_air_dir,
                        gt_dir=render_air_gt_dir,
                        fg_masks_by_stem=fg_masks_by_stem,
                        args=args,
                        prefix="AIR_",
                        compute_no_ref=True,
                    )
                    full_dict[scene_dir][method].update(air_metrics)

                if renders_air_bgblack_dir is not None:
                    air_bgblack_metrics, _ = _eval_rgb_set(
                        renders_dir=renders_air_bgblack_dir,
                        gt_dir=render_air_gt_dir,
                        fg_masks_by_stem=fg_masks_by_stem,
                        args=args,
                        prefix="AIRBGBLACK_",
                        compute_no_ref=True,
                    )
                    full_dict[scene_dir][method].update(air_bgblack_metrics)

                if depth_exr_dir.is_dir() and (gt_color_dir is not None):
                    depth_metrics_list = []
                    depth_fg_metrics_list = []
                    for name in image_names:
                        stem = os.path.splitext(name)[0]
                        pred_path = depth_exr_dir / f"{stem}.exr"
                        gt_path = Path(source_path_local) / "images" / "DepthImage" / f"{stem}.exr"
                        if pred_path.exists() and gt_path.exists():
                            if not args.foreground_only:
                                dm = eval_depth_one(
                                    str(pred_path), str(gt_path),
                                    pred_channel="Z", gt_channel="R",
                                    pred_scale=1000.0, gt_scale=1000.0,
                                    void_depth_raw=depth_void_value,
                                    void_eps=depth_void_eps,
                                    align="strict",
                                    eval_mask=None,
                                )
                                depth_metrics_list.append(dm)

                            if args.eval_foreground and (stem in fg_masks_by_stem):
                                dm_fg = eval_depth_one(
                                    str(pred_path), str(gt_path),
                                    pred_channel="Z", gt_channel="R",
                                    pred_scale=1000.0, gt_scale=1000.0,
                                    void_depth_raw=depth_void_value,
                                    void_eps=depth_void_eps,
                                    align="strict",
                                    eval_mask=fg_masks_by_stem[stem],
                                )
                                depth_fg_metrics_list.append(dm_fg)

                    if depth_metrics_list and (not args.foreground_only):
                        keys = sorted({k for d in depth_metrics_list for k in d.keys()})
                        depth_mean = {k: float(np.nanmean([d[k] for d in depth_metrics_list])) for k in keys}
                        full_dict[scene_dir][method].update(depth_mean)

                    if depth_fg_metrics_list and args.eval_foreground:
                        keys = sorted({k for d in depth_fg_metrics_list for k in d.keys()})
                        depth_fg_mean = {("FG_" + k): float(np.nanmean([d[k] for d in depth_fg_metrics_list])) for k in keys}
                        full_dict[scene_dir][method].update(depth_fg_mean)

                if normal_exr_dir.is_dir() and (gt_color_dir is not None):
                    normal_metrics_list = []
                    normal_fg_metrics_list = []
                    for name in image_names:
                        stem = os.path.splitext(name)[0]
                        pred_path = normal_exr_dir / f"{stem}.exr"
                        gt_path = Path(source_path_local) / "images" / "NormalImage" / f"{stem}.exr"
                        gt_depth_path = Path(source_path_local) / "images" / "DepthImage" / f"{stem}.exr"
                        if pred_path.exists() and gt_path.exists() and gt_depth_path.exists():
                            if not args.foreground_only:
                                nm = eval_normals_one(
                                    str(pred_path), str(gt_path),
                                    gt_depth_path=str(gt_depth_path),
                                    pred_enc="auto", gt_enc="auto",
                                    gt_depth_channel="R",
                                    ignore_depth_eq_value=1.0, ignore_depth_eps=1e-6,
                                    eval_mask=None,
                                )
                                normal_metrics_list.append(nm)

                            if args.eval_foreground and (stem in fg_masks_by_stem):
                                nm_fg = eval_normals_one(
                                    str(pred_path), str(gt_path),
                                    gt_depth_path=str(gt_depth_path),
                                    pred_enc="auto", gt_enc="auto",
                                    gt_depth_channel="R",
                                    ignore_depth_eq_value=1.0, ignore_depth_eps=1e-6,
                                    eval_mask=fg_masks_by_stem[stem],
                                )
                                normal_fg_metrics_list.append(nm_fg)

                    if normal_metrics_list and (not args.foreground_only):
                        keys = sorted({k for d in normal_metrics_list for k in d.keys()})
                        normal_mean = {k: float(np.nanmean([d[k] for d in normal_metrics_list])) for k in keys}
                        full_dict[scene_dir][method].update(normal_mean)

                    if normal_fg_metrics_list and args.eval_foreground:
                        keys = sorted({k for d in normal_fg_metrics_list for k in d.keys()})
                        normal_fg_mean = {("FG_" + k): float(np.nanmean([d[k] for d in normal_fg_metrics_list])) for k in keys}
                        full_dict[scene_dir][method].update(normal_fg_mean)

            with open(scene_dir + "/results.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True, ensure_ascii=False)

        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, "reason:", e)


if __name__ == "__main__":
    print(f"[Info] Using device: {DEVICE}")

    parser = ArgumentParser(description="UW_PGSR evaluation: RGB/Depth/Normal with optional Foreground metrics (render + render_air + render_air_bgblack + no-reference AIR metrics)")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    parser.add_argument("--source_path", "-s", required=False, type=str, default=None)
    parser.add_argument("--air_source_path", "-sa", required=False, type=str, default=None,
                        help="render_air / render_air_bgblack 对应 GT 路径。若未提供或为空，则自动从 source_path 推断：例如 /.../with/blue/4 -> /.../with/air。若推断失败且 method_dir/gt_air 也不存在，则仅评估 UIQM/UCIQE/NIQE。")

    parser.add_argument("--depth_void_value", type=float, default=1.0,
                        help="void 哨兵值（raw 深度 [0,1] 下），默认 1.0 表示 1000m 也即无限远")
    parser.add_argument("--depth_void_eps", type=float, default=1e-6,
                        help="void 判断容差（raw 值域下），默认 1e-6")

    parser.add_argument("--eval_foreground", action="store_true",
                        help="额外计算前景指标，前景由 renders_depth_exr 的深度图决定（render / render_air / render_air_bgblack 共用）")
    parser.add_argument("--foreground_only", action="store_true",
                        help="仅写入前景指标，不写全图 SSIM/PSNR/LPIPS，深度法线同理；无参考指标仍会写入")

    parser.add_argument("--fg_band_ratio", type=float, default=0.8,
                        help="主深度带覆盖的非 void 像素比例，建议 0.7 到 0.9")
    parser.add_argument("--fg_ratio_in_band", type=float, default=0.5,
                        help="在主深度带里取更近的比例作为前景，建议 0.3 到 0.7")
    parser.add_argument("--fg_hist_bins", type=int, default=256,
                        help="寻找主深度带时的直方图 bin 数")
    parser.add_argument("--fg_min_valid_px", type=int, default=1024,
                        help="有效非 void 像素太少则退化为仅用非 void 区作为前景")
    parser.add_argument("--fg_bbox_pad", type=int, default=8,
                        help="对前景 mask 的 bbox 做 padding，减小 SSIM 窗口边界效应")

    parser.add_argument("--nr_max_side", type=int, default=640,
                        help="无参考指标(UIQM/UCIQE/NIQE)计算前，把图像最长边缩放到该值以加速，<=0 表示不缩放")

    args = parser.parse_args()

    if args.depth_void_value is not None and args.depth_void_value > 2.0:
        print(f"[Info] depth_void_value={args.depth_void_value} 看起来不是 raw([0,1]) 值，自动改为 1.0")
        args.depth_void_value = 1.0

    evaluate(
        args.model_paths,
        args.source_path,
        args.air_source_path,
        depth_void_value=args.depth_void_value,
        depth_void_eps=args.depth_void_eps,
        args=args,
    )