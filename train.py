#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight, depth_aware_smooth_loss, anisotropic_total_variation_loss
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel
import scene.cameras as scene_cameras
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.medium_model import MediumEstimator
from scene.complete_physics_model import CompleteUnderwaterPhysics
from scene.cameras import Camera
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F
from lpipsPyTorch.modules.lpips import LPIPS

# -------------------- NEW: helper funcs for fake-air supervision / masking --------------------
def _safe_2d(x):
    """(H,W) or (1,H,W) -> (H,W)"""
    if x is None:
        return None
    return x.squeeze(0) if x.dim() == 3 else x

def _sobel_grad_mag(gray_1hw: torch.Tensor):
    """
    gray_1hw: [1,H,W], range [0,1]
    return: [1,H,W]
    """
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                      device=gray_1hw.device, dtype=gray_1hw.dtype).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                      device=gray_1hw.device, dtype=gray_1hw.dtype).view(1, 1, 3, 3) / 8.0
    x = gray_1hw.unsqueeze(0)  # [1,1,H,W]
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return g.squeeze(0)  # [1,H,W]

# -------------------- NEW: BG mask (GT + rendered_alpha + B_inf) aligned with tools/check_masks.py --------------------
@torch.no_grad()
def build_bg_mask_gt_alpha_binf(
    gt_3hw: torch.Tensor,
    gt_gray_1hw: torch.Tensor,
    alpha_1hw: torch.Tensor,
    *,
    binf_3: torch.Tensor = None,            # [3] or [3,1,1]
    nohit_hw: torch.Tensor = None,          # [H,W] in {0,1} or bool; if provided, multiplies final mask
    alpha_bg_thr: float = 0.10,
    alpha_gamma: float = 1.5,
    gt_w: float = 0.58,
    seed_gray_min: float = 0.03,
    seed_alpha_thr: float = 0.03,
    seed_binf_eps: float = 0.35,
    min_seed_px: int = 512,
    seed_std_thr: float = 0.08,
    binf_calib_w: float = 0.5,
    binf_eps: float = 0.25,
    use_binf_gate: bool = True,
    sigma_col_min: float = 0.02,
    sigma_col_max: float = 0.15,
    col_thr: float = 0.35,
    morph_k: int = 5,
):
    """
    Returns:
      bg_soft: [1,H,W] in [0,1] or None
      B_ref:   [3] (used ref color, binf calibrated toward per-view seed median) or None
      stats:   dict
    Notes:
      - This is a *no-connectivity* mask (robust & cheap), aligned with tools/check_masks.py.
      - For bg-consistency loss, pass nohit_hw to keep "no_hit" behavior (mask only where not hit).
      - For pruning, do NOT pass nohit_hw (we want to catch bad hits).
    """
    device = gt_3hw.device
    _, H, W = gt_3hw.shape
    a = alpha_1hw.squeeze(0) if alpha_1hw.dim() == 3 else alpha_1hw
    a = a.clamp(0.0, 1.0)
    gray = gt_gray_1hw.squeeze(0) if gt_gray_1hw.dim() == 3 else gt_gray_1hw
    gray = gray.clamp(0.0, 1.0)
    valid = (gray > float(seed_gray_min))

    eps = 1e-6
    gt_w = float(max(0.0, min(1.0, gt_w)))

    # alpha prior
    thr = float(max(alpha_bg_thr, 1e-6))
    w_alpha = ((thr - a) / thr).clamp(0.0, 1.0)
    w_alpha = (w_alpha ** float(max(alpha_gamma, 1e-6))) * valid.float()

    # binf formatting
    if binf_3 is not None and torch.is_tensor(binf_3):
        b = binf_3.detach()
        if b.numel() == 3:
            b = b.view(3, 1, 1)
        elif b.dim() == 3 and b.shape[0] == 3:
            b = b[:, :1, :1]
        else:
            b = None
    else:
        b = None

    # Seed selection
    if b is None:
        seed = valid & (a < min(float(seed_alpha_thr), float(alpha_bg_thr)))
        if seed.sum().item() < 64 and valid.any():
            seed = valid & (a < float(alpha_bg_thr))
        if seed.sum().item() < 64:
            seed = valid

        vals = gt_3hw[:, seed]
        if vals.numel() < 3:
            return None, None, {"skip": 1, "reason": "few_seed", "seed_pixels": int(seed.sum().item())}
        B_ref = torch.quantile(vals, 0.5, dim=1).clamp(0.0, 1.0).view(3, 1, 1)
    else:
        dist_binf = (gt_3hw - b.to(gt_3hw.dtype)).abs().mean(0)  # [H,W]
        seed = valid & (dist_binf < float(seed_binf_eps)) & (a < float(seed_alpha_thr))
        if seed.sum().item() < 64:
            seed = valid & (dist_binf < float(seed_binf_eps)) & (a < float(alpha_bg_thr))

        # ensure enough seed pixels by top-k sampling on score
        if seed.sum().item() < int(min_seed_px):
            cand = valid
            score = (1.0 - torch.clamp(dist_binf / max(float(seed_binf_eps), 1e-6), 0.0, 1.0)) * (1.0 - a)
            score = score * cand.float()
            flat_cand = cand.view(-1)
            idx = torch.nonzero(flat_cand, as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                k = min(int(min_seed_px), int(idx.numel()))
                top = torch.topk(score.view(-1)[idx], k=k, largest=True).indices
                pick = idx[top]
                seed_flat = torch.zeros((H * W,), device=device, dtype=torch.bool)
                seed_flat[pick] = True
                seed = seed_flat.view(H, W)

        n_seed = int(seed.sum().item())
        if n_seed < 32:
            return None, None, {"skip": 1, "reason": "few_seed", "seed_pixels": n_seed}

        vals = gt_3hw[:, seed]
        B_gt = torch.quantile(vals, 0.5, dim=1).clamp(0.0, 1.0)  # [3]
        std = float(torch.std(vals, dim=1).mean().item())
        if std > float(seed_std_thr):
            return None, None, {"skip": 1, "reason": "seed_std", "seed_pixels": n_seed, "seed_std": std}

        w = max(0.0, min(1.0, float(binf_calib_w)))
        B_ref = ((1.0 - w) * b.view(3) + w * B_gt).view(3, 1, 1).clamp(0.0, 1.0)

    # color weight
    dist = (gt_3hw - B_ref.to(gt_3hw.dtype)).abs().mean(0)  # [H,W]
    sigma = float(torch.std(gt_3hw[:, seed], dim=1).mean().item()) if ('seed' in locals() and seed.any()) else 0.05
    sigma = max(float(sigma_col_min), min(float(sigma_col_max), sigma))
    w_col = torch.exp(-(dist * dist) / (2.0 * sigma * sigma)).clamp(0.0, 1.0) * valid.float()
    if bool(use_binf_gate) and (b is not None):
        w_col = w_col * (dist < float(binf_eps)).float()

    bg = torch.exp(gt_w * torch.log(w_col + eps) + (1.0 - gt_w) * torch.log(w_alpha + eps))
    bg_soft = bg.unsqueeze(0).clamp(0.0, 1.0)

    k = int(morph_k)
    if k > 1:
        if k % 2 == 0:
            k += 1
        bg_soft = F.avg_pool2d(bg_soft.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0).clamp(0.0, 1.0)

    if nohit_hw is not None:
        nh = nohit_hw.squeeze(0) if (torch.is_tensor(nohit_hw) and nohit_hw.dim() == 3) else nohit_hw
        if torch.is_tensor(nh) and nh.shape == (H, W):
            nhf = nh.float() if nh.dtype != torch.float32 else nh
            bg_soft = (bg_soft * nhf.unsqueeze(0)).clamp(0.0, 1.0)

    stats = {
        "skip": 0,
        "mask_mean": float(bg_soft.mean().item()),
        "sigma_col": float(sigma),
        "gt_w": float(gt_w),
        "alpha_bg_thr": float(alpha_bg_thr),
        "alpha_gamma": float(alpha_gamma),
        "use_binf": int(1 if b is not None else 0),
    }
    return bg_soft, B_ref.view(3).detach(), stats

def _laplacian_rgb(rgb_3hw: torch.Tensor):
    """rgb_3hw: [3,H,W] -> [3,H,W]"""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                     device=rgb_3hw.device, dtype=rgb_3hw.dtype).view(1, 1, 3, 3)
    x = rgb_3hw.unsqueeze(0)  # [1,3,H,W]
    y = F.conv2d(x, k.repeat(3, 1, 1, 1), padding=1, groups=3)
    return y.squeeze(0)

def _multiscale_lap_loss(pred_3hw, tgt_3hw, mask_1hw, scales=(1, 2)):
    """
    pred/tgt: [3,H,W], mask: [1,H,W]
    """
    loss = 0.0
    for s in scales:
        if s > 1:
            pred = F.interpolate(pred_3hw.unsqueeze(0), scale_factor=1.0 / s, mode="bilinear", align_corners=False).squeeze(0)
            tgt = F.interpolate(tgt_3hw.unsqueeze(0), scale_factor=1.0 / s, mode="bilinear", align_corners=False).squeeze(0)
            m = F.interpolate(mask_1hw.unsqueeze(0), scale_factor=1.0 / s, mode="nearest").squeeze(0)
        else:
            pred, tgt, m = pred_3hw, tgt_3hw, mask_1hw

        lap_p = _laplacian_rgb(pred)
        lap_t = _laplacian_rgb(tgt)
        diff = torch.abs(lap_p - lap_t) * m
        denom = (m.sum() * 3.0) + 1e-6
        loss = loss + diff.sum() / denom
    return loss

# -------------------- Innovation 3: background mask + color (ROLLBACK to exp_bg_bz version) --------------------
def build_bg_mask_and_color(
    gt_3hw: torch.Tensor,
    gt_gray_1hw: torch.Tensor,
    depth_hw: torch.Tensor,
    alpha_1hw: torch.Tensor,
    hit_hw: torch.Tensor = None,
    *,
    border: int = 24,
    grad_thr: float = 0.02,
    z_q: float = 0.90,
    z_tau_ratio: float = 0.05,
    seed_top_q: float = 0.98,
    min_seed_ratio: float = 0.01,
    morph_k: int = 5,
    seed_std_thr: float = 0.05,
    sigma_col_min: float = 0.02,
    sigma_col_max: float = 0.12,
):
    """
    ROLLBACK (matches outputs/exp_bg_bz):
      - Background mask from: far * low-texture * no-hit * color-consistent-to-B_gt
      - Seeds chosen by top-quantile of (far * low-tex * no-hit) with border prior
      - Returns (mask_bg [1,H,W], B_gt [3], stats) or (None,None,stats) if unreliable.
    """
    device = gt_3hw.device
    _, H, W = gt_3hw.shape
    depth = depth_hw.squeeze(0) if depth_hw.dim() == 3 else depth_hw
    alpha = alpha_1hw.squeeze(0) if alpha_1hw.dim() == 3 else alpha_1hw

    # hit/no-hit: prefer explicit hit map if provided (more stable than alpha at edges)
    if hit_hw is not None:
        hit = hit_hw.squeeze(0) if hit_hw.dim() == 3 else hit_hw
        hit = (hit > 0).float()
    else:
        hit = (alpha > 1e-4).float()
    # Robustness: ensure hit map matches depth map shape; otherwise fall back to alpha-based hit.
    if hit.shape != depth.shape:
        hit = (alpha > 1e-4).float()

    # 1) low-texture region from GT grayscale gradient
    g = _sobel_grad_mag(gt_gray_1hw)  # [1,H,W]
    grad = g.squeeze(0)  # [H,W]
    low_tex = (grad < float(grad_thr)).float()

    # 2) border mask
    b = int(border)
    b = max(1, min(b, min(H // 2, W // 2)))
    border_w = torch.zeros((H, W), device=device, dtype=torch.float32)
    border_w[:b, :] = 1.0
    border_w[-b:, :] = 1.0
    border_w[:, :b] = 1.0
    border_w[:, -b:] = 1.0

    # 3) far threshold from depth quantile over hit pixels (robust)
    with torch.no_grad():
        valid = (depth > 0) & (hit > 0.5)
        if valid.any():
            z_cut = float(torch.quantile(depth[valid].float(), float(z_q)).item())
        else:
            return None, None, {"skip": 1, "reason": "no_valid_depth", "seed_pixels": 0}

    z_tau = max(float(z_tau_ratio) * float(z_cut), 1e-3)
    # far weight; treat invalid depth as far
    w_far = torch.sigmoid((depth - float(z_cut)) / float(z_tau)).clamp(0.0, 1.0)
    w_far = torch.where(depth <= 0, torch.ones_like(w_far), w_far)
    w_nohit = (1.0 - hit).clamp(0.0, 1.0)

    # Candidate score (soft): far * low-tex * no-hit, with border as a weak prior
    score = (w_far * low_tex * w_nohit).detach()
    score = score * (0.7 + 0.3 * border_w)

    with torch.no_grad():
        thr = float(torch.quantile(score.reshape(-1).float(), float(seed_top_q)).item())
    seed = score >= thr
    n_seed = int(seed.sum().item())
    if n_seed < int(float(min_seed_ratio) * H * W):
        return None, None, {"skip": 1, "reason": "few_seed", "seed_pixels": n_seed, "z_cut": float(z_cut)}

    # 5) estimate background color and self-check by std
    vals = gt_3hw[:, seed]  # [3,N]
    B_gt = torch.quantile(vals, 0.5, dim=1).clamp(0.0, 1.0)  # median
    std = float(torch.std(vals, dim=1).mean().item())
    if std > float(seed_std_thr):
        return None, None, {"skip": 1, "reason": "seed_std", "seed_pixels": n_seed, "seed_std": std, "z_cut": float(z_cut)}

    # 6) adaptive soft color consistency
    with torch.no_grad():
        sigma = float(torch.std(vals, dim=1).mean().item())
        sigma = max(float(sigma_col_min), min(float(sigma_col_max), sigma))
    dist = (gt_3hw - B_gt.view(3, 1, 1)).abs().mean(0)  # [H,W]
    w_col = torch.exp(-(dist * dist) / (2.0 * sigma * sigma)).clamp(0.0, 1.0)

    # final soft mask: far & nohit & lowtex & color-consistent
    mask = (w_far * w_nohit * low_tex * w_col).unsqueeze(0).clamp(0.0, 1.0)  # [1,H,W]

    # 7) soft morphology smoothing (avg pool) to reduce flicker / hard edges
    k = int(morph_k) if morph_k is not None else 0
    if k > 1:
        if k % 2 == 0:
            k += 1
        mask = F.avg_pool2d(mask.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0)
        mask = mask.clamp(0.0, 1.0)

    stats = {
        "seed_pixels": n_seed,
        "seed_std": std,
        "z_cut": float(z_cut),
        "z_tau": float(z_tau),
        "sigma_col": float(sigma),
        "mask_mean": float(mask.mean().item()),
        "skip": 0,
    }
    return mask, B_gt.detach(), stats

# -------------------- BG pruning helper: bg mask that can overlap with hits --------------------
def build_bg_mask_and_color_for_prune(
    gt_3hw: torch.Tensor,
    gt_gray_1hw: torch.Tensor,
    depth_hw: torch.Tensor,
    alpha_1hw: torch.Tensor,
    hit_hw: torch.Tensor = None,
    *,
    border: int = 24,
    grad_thr: float = 0.02,
    z_q: float = 0.90,
    z_tau_ratio: float = 0.05,
    seed_top_q: float = 0.98,
    min_seed_ratio: float = 0.01,
    morph_k: int = 5,
    seed_std_thr: float = 0.05,
    sigma_col_min: float = 0.02,
    sigma_col_max: float = 0.12,
):
    """
    Similar to build_bg_mask_and_color(), BUT:
      - Seeds are still selected from (far & low-tex & no-hit) for reliability (estimate B_gt)
      - The returned mask does NOT include w_nohit term, so it can overlap with "bad hits" (Gaussians covering bg).

    Returns: (mask_bg_prune [1,H,W], B_gt [3], stats) or (None,None,stats) if unreliable.
    """
    device = gt_3hw.device
    _, H, W = gt_3hw.shape
    depth = depth_hw.squeeze(0) if depth_hw.dim() == 3 else depth_hw
    alpha = alpha_1hw.squeeze(0) if alpha_1hw.dim() == 3 else alpha_1hw
    if hit_hw is not None:
        hit = hit_hw.squeeze(0) if hit_hw.dim() == 3 else hit_hw
        hit = (hit > 0).float()
    else:
        hit = (alpha > 1e-4).float()
    if hit.shape != depth.shape:
        hit = (alpha > 1e-4).float()

    g = _sobel_grad_mag(gt_gray_1hw)  # [1,H,W]
    grad = g.squeeze(0)
    low_tex = (grad < float(grad_thr)).float()

    b = int(border)
    b = max(1, min(b, min(H // 2, W // 2)))
    border_w = torch.zeros((H, W), device=device, dtype=torch.float32)
    border_w[:b, :] = 1.0
    border_w[-b:, :] = 1.0
    border_w[:, :b] = 1.0
    border_w[:, -b:] = 1.0

    with torch.no_grad():
        valid = (depth > 0) & (hit > 0.5)
        if valid.any():
            z_cut = float(torch.quantile(depth[valid].float(), float(z_q)).item())
        else:
            return None, None, {"skip": 1, "reason": "no_valid_depth", "seed_pixels": 0}

    z_tau = max(float(z_tau_ratio) * float(z_cut), 1e-3)
    w_far = torch.sigmoid((depth - float(z_cut)) / float(z_tau)).clamp(0.0, 1.0)
    w_far = torch.where(depth <= 0, torch.ones_like(w_far), w_far)
    w_nohit = (1.0 - hit).clamp(0.0, 1.0)

    # Seeds: far * low-tex * no-hit, with border prior
    score = (w_far * low_tex * w_nohit).detach()
    score = score * (0.7 + 0.3 * border_w)
    with torch.no_grad():
        thr = float(torch.quantile(score.reshape(-1).float(), float(seed_top_q)).item())
    seed = score >= thr
    n_seed = int(seed.sum().item())
    if n_seed < int(float(min_seed_ratio) * H * W):
        return None, None, {"skip": 1, "reason": "few_seed", "seed_pixels": n_seed, "z_cut": float(z_cut)}

    vals = gt_3hw[:, seed]
    B_gt = torch.quantile(vals, 0.5, dim=1).clamp(0.0, 1.0)
    std = float(torch.std(vals, dim=1).mean().item())
    if std > float(seed_std_thr):
        return None, None, {"skip": 1, "reason": "seed_std", "seed_pixels": n_seed, "seed_std": std, "z_cut": float(z_cut)}

    with torch.no_grad():
        sigma = float(torch.std(vals, dim=1).mean().item())
        sigma = max(float(sigma_col_min), min(float(sigma_col_max), sigma))
    dist = (gt_3hw - B_gt.view(3, 1, 1)).abs().mean(0)
    w_col = torch.exp(-(dist * dist) / (2.0 * sigma * sigma)).clamp(0.0, 1.0)

    # NOTE: no w_nohit here, so mask can overlap with hits (bad Gaussians covering bg)
    mask = (w_far * low_tex * w_col).unsqueeze(0).clamp(0.0, 1.0)

    k = int(morph_k) if morph_k is not None else 0
    if k > 1:
        if k % 2 == 0:
            k += 1
        mask = F.avg_pool2d(mask.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0)
        mask = mask.clamp(0.0, 1.0)

    stats = {
        "seed_pixels": n_seed,
        "seed_std": std,
        "z_cut": float(z_cut),
        "z_tau": float(z_tau),
        "sigma_col": float(sigma),
        "mask_mean": float(mask.mean().item()),
        "skip": 0,
        "note": "prune_mask_nohitless",
    }
    return mask, B_gt.detach(), stats

def setup_seed(seed):
     # Make training as reproducible as possible across runs.
     # NOTE: Full determinism is not guaranteed due to some CUDA ops, but this removes the biggest source
     # of run-to-run drift (CUDA RNG / init).
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
def _maybe_enable_determinism(enable: bool):
    """
    Optional stricter determinism. This may slow training and/or warn about unsupported ops.
    We keep this OFF by default.
    """
    if not enable:
        return
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Keep training robust even if some torch versions don't support this.
        pass

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(cam.colmap_id, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, cam.uid,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        ncc_scale=cam.ncc_scale,
                        preload_img=False, data_device = cam.data_device)
    return virtul_cam

def water_aware_densify_and_prune(gaussians, render_pkg, opt, scene_extent, size_threshold, 
                                 water_depth_threshold=3.0, water_gradient_scale=0.1):
    """
    Water-aware densification and pruning to prevent 3DGS from modeling water effects.
    
    Args:
        gaussians: GaussianModel instance
        render_pkg: Render package containing depth information
        opt: Optimization parameters
        scene_extent: Scene extent for pruning
        size_threshold: Size threshold for pruning
        water_depth_threshold: Depth threshold to identify water regions
        water_gradient_scale: Scale factor for gradients in water regions
    """
    # Get depth information
    depth_map = render_pkg.get('plane_depth', None)
    if depth_map is None:
        # Fallback to standard densification if no depth info
        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                   opt.opacity_cull_threshold, scene_extent, size_threshold)
        return

# -------------------- BG mask pruning (aligned with tools/check_masks.py) --------------------
def _bg_mask_prune(
    gaussians,
    render_pkg,
    viewpoint_cam,
    gt_image_3hw: torch.Tensor,
    gt_gray_1hw: torch.Tensor,
    pipe,
    opt,
    iteration: int,
    tb_writer=None,
):
    """
    UPDATED pruning trigger (no more double evidence):
      target_2d = BG_MASK(gt + rendered_alpha + B_inf)  AND  (rendered_alpha > bg_prune_presence_thr)
    i.e. prune Gaussians that cover pixels that *should be background*.

    We keep the robust mapping & accumulation logic (dilation pyramid + center sampling + evidence over time),
    but the target definition is now aligned with tools/check_masks.py.
    """
    if not getattr(opt, "bg_prune_enable", False):
        return
    if iteration < int(getattr(opt, "bg_prune_from_iter", 12000)):
        return
    until = int(getattr(opt, "bg_prune_until_iter", -1))
    if until > 0 and iteration > until:
        return
    interval = int(getattr(opt, "bg_prune_interval", 100))
    if interval <= 0 or (iteration % interval) != 0:
        return
    if not bool(getattr(pipe, "use_underwater_model", False)):
        return
    if ("rendered_alpha" not in render_pkg) or ("viewspace_points_abs" not in render_pkg):
        return

    dbg = bool(int(getattr(opt, "bg_prune_debug", 0)) == 1)

    try:
        # Store last prune count for console/tqdm display.
        if not hasattr(gaussians, "_bg_prune_last_num"):
            gaussians._bg_prune_last_num = 0
        gaussians._bg_prune_last_num = 0
        with torch.no_grad():
            alpha2d = render_pkg["rendered_alpha"].squeeze(0).clamp(0.0, 1.0)
            hit_map = render_pkg.get("hit_map", None)
            # Build BG mask (no no-hit here; we want to catch bad hits)
            binf_3 = None
            try:
                if complete_physics_model is not None and hasattr(complete_physics_model, "backscatter_net") and hasattr(complete_physics_model.backscatter_net, "B_inf"):
                    binf_3 = complete_physics_model.backscatter_net.B_inf.detach().view(3)
            except Exception:
                binf_3 = None

            bg_soft, _, st = build_bg_mask_gt_alpha_binf(
                gt_image_3hw, gt_gray_1hw, alpha2d,
                binf_3=binf_3,
                nohit_hw=None,
                alpha_bg_thr=float(getattr(opt, "bgm_alpha_bg_thr", 0.10)),
                alpha_gamma=float(getattr(opt, "bgm_alpha_gamma", 1.5)),
                gt_w=float(getattr(opt, "bgm_gt_w", 0.58)),
                seed_gray_min=float(getattr(opt, "bgm_seed_gray_min", 0.03)),
                seed_alpha_thr=float(getattr(opt, "bgm_seed_alpha_thr", 0.03)),
                seed_binf_eps=float(getattr(opt, "bgm_seed_binf_eps", 0.35)),
                min_seed_px=int(getattr(opt, "bgm_min_seed_px", 512)),
                seed_std_thr=float(getattr(opt, "bgm_seed_std_thr", 0.08)),
                binf_calib_w=float(getattr(opt, "bgm_binf_calib_w", 0.5)),
                binf_eps=float(getattr(opt, "bgm_binf_eps", 0.25)),
                use_binf_gate=bool(int(getattr(opt, "bgm_use_binf_gate", 1)) == 1),
                sigma_col_min=float(getattr(opt, "bgm_sigma_col_min", 0.02)),
                sigma_col_max=float(getattr(opt, "bgm_sigma_col_max", 0.15)),
                col_thr=float(getattr(opt, "bgm_col_thr", 0.35)),
                morph_k=int(getattr(opt, "bgm_morph_k", 5)),
            )
            if bg_soft is None or st is None or int(st.get("skip", 1)) != 0:
                if dbg:
                    try:
                        reason = None if st is None else st.get("reason", "unknown")
                        print(f"[ITER {iteration}] BG-prune skip: bg_mask unavailable (reason={reason})")
                    except Exception:
                        pass
                return

            m_thr = float(getattr(opt, "bg_prune_mask_thr", 0.25))
            presence_thr = float(getattr(opt, "bg_prune_presence_thr", 1e-4))
            target = ((bg_soft.squeeze(0) > m_thr) & (alpha2d > presence_thr)).float()
            if target.sum().item() < 1:
                if dbg:
                    try:
                        print(f"[ITER {iteration}] BG-prune skip: target empty (mask_thr={m_thr}, presence_thr={presence_thr})")
                    except Exception:
                        pass
                return
            target_ratio = float((target > 0).float().mean().item())
            if target_ratio < float(getattr(opt, "bg_prune_mask_ratio_min", 0.002)) or \
               target_ratio > float(getattr(opt, "bg_prune_mask_ratio_max", 0.40)):
                if dbg:
                    try:
                        rmin = float(getattr(opt, "bg_prune_mask_ratio_min", 0.002))
                        rmax = float(getattr(opt, "bg_prune_mask_ratio_max", 0.40))
                        print(f"[ITER {iteration}] BG-prune skip: target_ratio={target_ratio:.6f} not in [{rmin:.6f},{rmax:.6f}]")
                    except Exception:
                        pass
                return

            # Build dilation pyramid (bool-ish in [0,1])
            x = target.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            D0 = x
            D1 = torch.nn.functional.max_pool2d(x, kernel_size=7, stride=1, padding=3)
            D2 = torch.nn.functional.max_pool2d(x, kernel_size=15, stride=1, padding=7)
            D3 = torch.nn.functional.max_pool2d(x, kernel_size=31, stride=1, padding=15)
            D0f = D0.view(-1)
            D1f = D1.view(-1)
            D2f = D2.view(-1)
            D3f = D3.view(-1)

            H, W = target.shape

            # 2D centers from rasterizer (abs coords)
            pts2d = render_pkg["viewspace_points_abs"]  # [N,3]
            if pts2d is None or pts2d.numel() == 0:
                return
            xs = pts2d[:, 0]
            ys = pts2d[:, 1]
            # If coords look like NDC, convert to pixel
            if (xs.abs().max().item() <= 2.0) and (ys.abs().max().item() <= 2.0):
                xs = (xs + 1.0) * 0.5 * float(W - 1)
                ys = (ys + 1.0) * 0.5 * float(H - 1)

            u = torch.clamp(xs.round().long(), 0, W - 1)
            v = torch.clamp(ys.round().long(), 0, H - 1)
            lin = v * W + u  # [N]

            radii = render_pkg.get("radii", None)
            if radii is None:
                return
            vis = render_pkg.get("visibility_filter", (radii > 0))
            if vis is None:
                vis = (radii > 0)
            vis = vis & (radii > 0)

            # Per-gaussian safety gates: edge, opacity (+ optional size/isolation)
            xyz = gaussians.get_xyz
            wv = viewpoint_cam.world_view_transform

            op = gaussians.get_opacity.squeeze()
            op_thr = float(getattr(opt, "bg_prune_opacity_thr", 0.6))
            op_gate = (op < op_thr)

            grad_map = _sobel_grad_mag(gt_gray_1hw).squeeze(0)  # [H,W]
            edge_thr = float(getattr(opt, "bg_prune_edge_thr", 0.05))
            edge_gate = (grad_map.view(-1)[lin] < edge_thr)

            # Extra heuristics:
            # - Large size gate (in screen-space radius)
            use_size_gate = bool(getattr(opt, "bg_prune_use_size_gate", True))
            min_r = float(getattr(opt, "bg_prune_min_radii", 16.0))
            r_gate = (radii.float() >= min_r) if use_size_gate else torch.ones_like(op_gate, dtype=torch.bool)

            # - Isolation gate: coarse 2D cell occupancy (large floaters tend to be isolated)
            use_iso = bool(getattr(opt, "bg_prune_use_isolation", True))
            iso_gate = torch.ones_like(op_gate, dtype=torch.bool)
            if use_iso:
                try:
                    grid = int(getattr(opt, "bg_prune_iso_grid", 32))
                    grid = max(grid, 4)
                    max_cnt = int(getattr(opt, "bg_prune_iso_max_count", 2))
                    cx = torch.clamp((u // grid), 0, max((W - 1) // grid, 0))
                    cy = torch.clamp((v // grid), 0, max((H - 1) // grid, 0))
                    n_cx = max((W - 1) // grid + 1, 1)
                    n_cy = max((H - 1) // grid + 1, 1)
                    cell = cy * n_cx + cx
                    cell = cell.long()
                    counts = torch.zeros((n_cx * n_cy,), device=cell.device, dtype=torch.int32)
                    counts.scatter_add_(0, cell[vis], torch.ones_like(cell[vis], dtype=torch.int32))
                    cell_cnt = counts[cell].to(torch.int32)
                    iso_gate = (cell_cnt <= max_cnt)
                except Exception:
                    iso_gate = torch.ones_like(op_gate, dtype=torch.bool)

            base = vis & op_gate & edge_gate & r_gate & iso_gate
            if base.sum().item() == 0:
                if dbg:
                    try:
                        print(f"[ITER {iteration}] BG-prune skip: no base candidates after safety gates")
                    except Exception:
                        pass
                return

            # Overlap score via radius bucket -> appropriate dilation
            s = torch.zeros_like(op, dtype=torch.float32)
            r = radii.float()
            idx0 = base & (r < 4)
            idx1 = base & (r >= 4) & (r < 8)
            idx2 = base & (r >= 8) & (r < 16)
            idx3 = base & (r >= 16)
            if idx0.any():
                s[idx0] = D0f[lin[idx0]]
            if idx1.any():
                s[idx1] = D1f[lin[idx1]]
            if idx2.any():
                s[idx2] = D2f[lin[idx2]]
            if idx3.any():
                s[idx3] = D3f[lin[idx3]]

            cov_thr = float(getattr(opt, "bg_prune_cov_thr", 0.5))
            cov_gate = (s > cov_thr)
            cand = base & cov_gate

            # Accumulate evidence (seen: visible; hit: candidate)
            if (not hasattr(gaussians, "bg_prune_seen")) or (gaussians.bg_prune_seen.numel() == 0):
                gaussians.bg_prune_seen = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
            if (not hasattr(gaussians, "bg_prune_hit")) or (gaussians.bg_prune_hit.numel() == 0):
                gaussians.bg_prune_hit = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

            gaussians.bg_prune_seen[vis] += 1.0
            gaussians.bg_prune_hit[cand] += 1.0

            seen = gaussians.bg_prune_seen.squeeze(1)
            hit = gaussians.bg_prune_hit.squeeze(1)
            ratio = hit / (seen + 1e-6)
            seen_min = int(getattr(opt, "bg_prune_seen_min", 5))
            ratio_thr = float(getattr(opt, "bg_prune_ratio_thr", 0.7))

            prune_candidates = (seen >= seen_min) & (ratio >= ratio_thr)
            if prune_candidates.sum().item() == 0:
                if tb_writer is not None and (iteration % 500 == 0):
                    tb_writer.add_scalar("bg_prune/target_ratio", target_ratio, iteration)
                    tb_writer.add_scalar("bg_prune/prune_num", 0, iteration)
                if dbg:
                    try:
                        print(f"[ITER {iteration}] BG-prune skip: no prune_candidates (seen_min={seen_min}, ratio_thr={ratio_thr:.3f}, cand={int(cand.sum().item())})")
                    except Exception:
                        pass
                return

            max_k = int(getattr(opt, "bg_prune_max_per_iter", 800))
            idx = torch.where(prune_candidates)[0]
            if idx.numel() > max_k:
                # prune highest ratio first
                vals = ratio[idx]
                topk = torch.topk(vals, k=max_k, largest=True).indices
                idx = idx[topk]

            prune_mask = torch.zeros_like(prune_candidates, dtype=torch.bool)
            prune_mask[idx] = True

            if prune_mask.sum().item() > 0:
                gaussians._bg_prune_last_num = int(prune_mask.sum().item())
                gaussians.prune_points(prune_mask)
                if iteration % 1 == 0:
                    print(f"[ITER {iteration}] BG-mask pruning removed {gaussians._bg_prune_last_num} points")

            if tb_writer is not None:
                tb_writer.add_scalar("bg_prune/target_ratio", target_ratio, iteration)
                tb_writer.add_scalar("bg_prune/cand_num", int(cand.sum().item()), iteration)
                tb_writer.add_scalar("bg_prune/prune_num", int(prune_mask.sum().item()), iteration)
    except Exception:
        # Keep training robust; never crash training due to pruning heuristics.
        return

# Backward-compatible alias (older training loops call this name).
def _bg_double_evidence_prune(*args, **kwargs):
    return _bg_mask_prune(*args, **kwargs)
    
    # Compute gradients
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom_abs
    grads[grads.isnan()] = 0.0
    grads_abs[grads_abs.isnan()] = 0.0
    max_radii2D = gaussians.max_radii2D.clone()
    
    # Get 3DGS point depths
    points_xyz = gaussians.get_xyz
    if points_xyz.shape[0] > 0:
        # Project 3DGS points to depth map to get their depths
        # This is a simplified approach - in practice, you might want more sophisticated depth estimation
        points_depth = torch.norm(points_xyz, dim=1)  # Approximate depth as distance from origin
        
        # Identify water regions
        water_mask = points_depth > water_depth_threshold
        
        # Scale down gradients in water regions to reduce densification
        if water_mask.any():
            grads[water_mask] *= water_gradient_scale
            grads_abs[water_mask] *= water_gradient_scale
    
    # Apply densification with modified gradients
    gaussians.densify_and_clone(grads, opt.densify_grad_threshold, scene_extent)
    gaussians.densify_and_split(grads, opt.densify_grad_threshold, grads_abs, opt.densify_abs_grad_threshold, 
                               scene_extent, max_radii2D)
    
    # Standard pruning
    prune_mask = (gaussians.get_opacity < opt.opacity_cull_threshold).squeeze()
    if size_threshold:
        big_points_vs = gaussians.max_radii2D > size_threshold
        big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene_extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()

def complete_physics_aware_densify_and_prune(gaussians, render_pkg, opt, scene_extent, size_threshold, pipe):
    """
    Complete physics-aware densification and pruning to ensure 3DGS only learns geometry.
    
    Args:
        gaussians: GaussianModel instance
        render_pkg: Render package containing physics components
        opt: Optimization parameters
        scene_extent: Scene extent for pruning
        size_threshold: Size threshold for pruning
        pipe: Pipeline parameters (包含可配置的超参数)
    """
    # Get physics components
    backscatter = render_pkg.get('backscatter', None)
    attenuation = render_pkg.get('attenuation', None)
    depth_map = render_pkg.get('plane_depth', None)
    
    if depth_map is None:
        # Fallback to standard densification if no depth info
        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                   opt.opacity_cull_threshold, scene_extent, size_threshold)
        return
    
    # Compute gradients
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom_abs
    grads[grads.isnan()] = 0.0
    grads_abs[grads_abs.isnan()] = 0.0
    max_radii2D = gaussians.max_radii2D.clone()
    
    # Get 3DGS point depths and physics constraints
    points_xyz = gaussians.get_xyz
    if points_xyz.shape[0] > 0:
        # Approximate depth for each 3DGS point (legacy behavior)
        points_depth = torch.norm(points_xyz, dim=1)
        
        # 使用可配置的深度阈值
        physics_region_threshold = getattr(pipe, 'physics_region_depth_threshold', 2.0)
        water_region_threshold = getattr(pipe, 'water_region_depth_threshold', 1.5)
        physics_region_scale = getattr(pipe, 'physics_region_gradient_scale', 0.1)
        water_region_scale = getattr(pipe, 'water_region_gradient_scale', 0.05)
        physics_strength_threshold = getattr(pipe, 'physics_strength_threshold', 0.3)
        
        # Identify regions where physics model should handle effects
        physics_region_mask = points_depth > physics_region_threshold
        
        # Scale down gradients in physics regions (可配置的缩放系数)
        if physics_region_mask.any():
            grads[physics_region_mask] *= physics_region_scale
            grads_abs[physics_region_mask] *= physics_region_scale
        
        # Additional constraint: if physics components are strong, reduce 3DGS activity
        if backscatter is not None and attenuation is not None:
            # Check if physics model is handling the effects well
            physics_strength = (backscatter.mean() + (1 - attenuation.mean())) / 2
            
            if physics_strength > physics_strength_threshold:  # 使用可配置的阈值
                # Further reduce 3DGS activity in water regions
                water_mask = points_depth > water_region_threshold
                if water_mask.any():
                    grads[water_mask] *= water_region_scale  # 使用可配置的缩放系数
                    grads_abs[water_mask] *= water_region_scale
    
    # Apply densification with modified gradients
    gaussians.densify_and_clone(grads, opt.densify_grad_threshold, scene_extent)
    gaussians.densify_and_split(grads, opt.densify_grad_threshold, grads_abs, opt.densify_abs_grad_threshold, 
                               scene_extent, max_radii2D)
    
    # Standard pruning
    prune_mask = (gaussians.get_opacity < opt.opacity_cull_threshold).squeeze()
    if size_threshold:
        big_points_vs = gaussians.max_radii2D > size_threshold
        big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene_extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    
    # ===== Depth-Aware Aggressive Pruning (inspired by 3D-UIR) =====
    # Get configurable pruning parameters
    far_prune_distance = getattr(opt, 'far_prune_distance', 20.0)  # 远距离剪枝阈值(米)
    far_opacity_threshold = getattr(opt, 'far_opacity_threshold', 0.3)  # 远距离不透明度阈值
    aggressive_prune = getattr(opt, 'aggressive_far_prune', True)  # 是否启用激进剪枝
    
    if aggressive_prune:
        # CRITICAL: Recompute points_depth after densification (point count has changed!)
        points_xyz_updated = gaussians.get_xyz
        if points_xyz_updated.shape[0] > 0:
            points_depth_updated = torch.norm(points_xyz_updated, dim=1)
            
            # Strategy 1: Aggressively prune far + weak points
            # Points that are far away AND have low opacity are likely modeling water/fog
            far_mask = points_depth_updated > far_prune_distance
            weak_mask = (gaussians.get_opacity < far_opacity_threshold).squeeze()
            far_weak_mask = torch.logical_and(far_mask, weak_mask)
            prune_mask = torch.logical_or(prune_mask, far_weak_mask)
            
            # Strategy 2: Prune very far points regardless of opacity (extreme distance)
            very_far_threshold = getattr(opt, 'very_far_prune_distance', 50.0)
            very_far_mask = points_depth_updated > very_far_threshold
            prune_mask = torch.logical_or(prune_mask, very_far_mask)
            
            # Strategy 3: If physics model is strong, aggressively prune water regions
            if backscatter is not None and attenuation is not None:
                physics_strength = (backscatter.mean() + (1 - attenuation.mean())) / 2
                if physics_strength > 0.4:  # Physics is handling water well
                    # Prune medium-far + weak points (more aggressive when physics is strong)
                    medium_far_mask = points_depth_updated > 10.0
                    medium_weak_mask = (gaussians.get_opacity < 0.5).squeeze()
                    medium_prune = torch.logical_and(medium_far_mask, medium_weak_mask)
                    prune_mask = torch.logical_or(prune_mask, medium_prune)
    
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()

def layered_l1_loss(pred_image, gt_image, render_pkg, water_depth_threshold=3.0, water_weight=0.1):
    """
    Layered L1 loss that applies different weights to scene and water regions.
    
    Args:
        pred_image: Predicted underwater image
        gt_image: Ground truth underwater image
        render_pkg: Render package containing depth information
        water_depth_threshold: Depth threshold to identify water regions
        water_weight: Weight for water regions (lower = less emphasis on water)
    
    Returns:
        Layered L1 loss
    """
    # Get depth information
    depth_map = render_pkg.get('plane_depth', None)
    if depth_map is None:
        # Fallback to standard L1 loss if no depth info
        return l1_loss(pred_image, gt_image)
    
    # Create water mask based on depth
    water_mask = depth_map > water_depth_threshold
    scene_mask = ~water_mask
    
    # Compute L1 loss for scene regions (full weight)
    scene_loss = l1_loss(pred_image * scene_mask, gt_image * scene_mask)
    
    # Compute L1 loss for water regions (reduced weight)
    water_loss = l1_loss(pred_image * water_mask, gt_image * water_mask) * water_weight
    
    # Combine losses
    total_loss = scene_loss + water_loss
    
    return total_loss

def physics_constraints(a_map, B_map, render_pkg, depth_weight=0.05, monotonic_weight=0.05):
    """
    Physics-based constraints for underwater imaging model.
    
    Args:
        a_map: Direct attenuation map A(z)
        B_map: Backscatter map B(z)
        render_pkg: Render package containing depth information
        depth_weight: Weight for depth-based constraints
        monotonic_weight: Weight for monotonic constraints
    
    Returns:
        Physics constraint loss
    """
    # Prefer the exact depth used by the physics model (metric depth z in meters)
    depth_map = render_pkg.get('physics_depth', None)
    if depth_map is None:
        depth_map = render_pkg.get('plane_depth', None)
    if depth_map is None:
        return torch.tensor(0.0, device=a_map.device)
    
    # Use metric depth z (meters). Do NOT invert depth here.
    depth_z = depth_map
    if depth_z.dim() == 3:
        depth_z = depth_z.squeeze(0)
    
    # Constraint 1: A(z) should decrease with depth (monotonic attenuation)
    # A(z) = e^(-βDz) should be monotonically decreasing
    if a_map.shape[1] > 1 and depth_z.shape[0] > 1:  # Check spatial dimensions
        # 计算深度梯度（带方向）
        depth_grad = depth_z[1:, :] - depth_z[:-1, :]
        
        # 计算衰减梯度（带方向）
        a_grad = a_map[:, 1:, :] - a_map[:, :-1, :]
        
        # 单调性约束：深度增加时，衰减应该减小
        # 即：depth_grad > 0 时，a_grad < 0
        # 违反单调性的情况：a_grad * depth_grad > 0
        monotonic_violation = torch.clamp(a_grad * depth_grad, min=0.0)
        monotonic_loss = monotonic_weight * torch.mean(monotonic_violation)
    else:
        monotonic_loss = torch.tensor(0.0, device=a_map.device)
    
    # Constraint 2: B(z) should increase with depth (more backscatter at greater depth)
    # B(z) = B∞•(1 – e^(-βBz)) should be monotonically increasing
    if B_map.shape[1] > 1 and depth_z.shape[0] > 1:
        # 计算深度梯度（带方向）
        depth_grad = depth_z[1:, :] - depth_z[:-1, :]
        
        # 计算后向散射梯度（带方向）
        B_grad = B_map[:, 1:, :] - B_map[:, :-1, :]
        
        # 单调性约束：深度增加时，后向散射应该增加
        # 即：depth_grad > 0 时，B_grad > 0
        # 违反单调性的情况：-B_grad * depth_grad > 0
        backscatter_violation = torch.clamp(-B_grad * depth_grad, min=0.0)
        backscatter_loss = monotonic_weight * torch.mean(backscatter_violation)
    else:
        backscatter_loss = torch.tensor(0.0, device=a_map.device)
    
    # Constraint 3: At infinite depth, A(z) → 0 and B(z) → B∞
    # This is naturally enforced by the exponential functions
    
    # Constraint 4: Energy conservation - A(z) + B(z) should not exceed 1
    energy_violation = torch.clamp(a_map + B_map - 1.0, min=0.0)
    energy_loss = depth_weight * energy_violation.mean()
    
    total_physics_loss = monotonic_loss + backscatter_loss + energy_loss
    return total_physics_loss

def dro_guided_water_pruning(gaussians, render_pkg, viewpoint_cam, opt, iteration, tb_writer=None):
    """
    使用DRO深度图指导水体雾墙剪枝 (混合方案C)
    
    核心思路：
      1. 对比PGSR深度 vs DRO参考深度
      2. 识别PGSR建模过近的点（可能是水体雾墙）
      3. 直接剪枝这些点，强制UIFM解耦水体
      4. 不参与梯度优化，避免与多视图冲突
    
    Args:
        gaussians: GaussianModel instance
        render_pkg: 渲染包，包含plane_depth
        viewpoint_cam: 当前相机，包含invdepthmap
        opt: 优化参数
        iteration: 当前迭代次数
    
    Returns:
        pruned_count: 剪枝的点数量
    """
    # 检查是否启用DRO剪枝
    if not getattr(opt, 'dro_guided_pruning', False):
        return 0
    
    # 检查深度可靠性
    if not hasattr(viewpoint_cam, 'depth_reliable') or not viewpoint_cam.depth_reliable:
        return 0
    
    if not hasattr(viewpoint_cam, 'invdepthmap') or viewpoint_cam.invdepthmap is None:
        return 0
    
    # 检查是否在激活范围内
    dro_prune_from_iter = getattr(opt, 'dro_prune_from_iter', 7000)
    dro_prune_until_iter = getattr(opt, 'dro_prune_until_iter', 15000)
    dro_prune_interval = getattr(opt, 'dro_prune_interval', 100)
    
    if iteration < dro_prune_from_iter or iteration > dro_prune_until_iter:
        return 0
    
    # 按间隔执行
    if iteration % dro_prune_interval != 0:
        return 0
    
    # Step 1: 获取深度信息
    if 'plane_depth' not in render_pkg:
        return 0
    
    pgsr_depth = render_pkg['plane_depth']  # (H, W) or (1, H, W)
    if pgsr_depth.dim() == 3:
        pgsr_depth = pgsr_depth.squeeze(0)  # (H, W)
    
    dro_inv_depth = viewpoint_cam.invdepthmap.cuda()  # (1, H, W)
    if dro_inv_depth.dim() == 3:
        dro_inv_depth = dro_inv_depth.squeeze(0)  # (H, W)
    
    # 转换为真实深度（米）
    dro_depth = 1.0 / (dro_inv_depth + 1e-5)  # (H, W)
    
    H, W = pgsr_depth.shape
    
    # Step 2: 计算深度差异（归一化到相同尺度）
    # 使用百分位归一化避免尺度问题
    try:
        pgsr_valid = pgsr_depth[pgsr_depth > 0]
        dro_valid = dro_depth[dro_depth > 0]
        
        if pgsr_valid.numel() < 100 or dro_valid.numel() < 100:
            return 0
        
        pgsr_p5 = torch.quantile(pgsr_valid, 0.05)
        pgsr_p95 = torch.quantile(pgsr_valid, 0.95)
        dro_p5 = torch.quantile(dro_valid, 0.05)
        dro_p95 = torch.quantile(dro_valid, 0.95)
        
        pgsr_depth_norm = (pgsr_depth - pgsr_p5) / (pgsr_p95 - pgsr_p5 + 1e-8)
        dro_depth_norm = (dro_depth - dro_p5) / (dro_p95 - dro_p5 + 1e-8)
    except:
        return 0
    
    # Step 3: 识别"雾墙"区域
    # PGSR深度显著小于DRO深度 → 可能是水体雾墙
    depth_diff_norm = pgsr_depth_norm - dro_depth_norm
    
    # 动态阈值：根据训练阶段线性调整（建议3）
    # 在 [dro_prune_from_iter, dro_prune_until_iter] 上，
    # 将阈值从 dro_depth_diff_threshold 逐渐放宽到 0.5 * dro_depth_diff_threshold（例如 -0.3 → -0.15）
    dro_depth_diff_threshold = getattr(opt, 'dro_depth_diff_threshold', -0.3)
    dro_prune_from_iter = getattr(opt, 'dro_prune_from_iter', 5000)
    dro_prune_until_iter = getattr(opt, 'dro_prune_until_iter', 15000)
    if dro_prune_until_iter > dro_prune_from_iter:
        progress = (iteration - dro_prune_from_iter) / float(dro_prune_until_iter - dro_prune_from_iter)
        progress = max(0.0, min(1.0, progress))  # clamp to [0, 1]
    else:
        progress = 0.0
    # 早期使用更严格的阈值，后期逐渐放宽，避免过晚阶段过度剪枝
    threshold = dro_depth_diff_threshold * (1.0 - 0.5 * progress)
    
    # 最大深度限制
    dro_prune_max_depth = getattr(opt, 'dro_prune_max_depth', 5.0)
    
    # 多重条件：确保是雾墙而非正常物体
    water_fog_mask = ((depth_diff_norm < threshold) &      # 深度显著偏近
                      (pgsr_depth < dro_prune_max_depth) & # 绝对深度很近
                      (pgsr_depth > 0.1))                  # 有效深度值
    
    if water_fog_mask.sum() == 0:
        return 0
    
    # Step 4: 将2D mask映射到3D高斯点
    points_xyz = gaussians.get_xyz
    if points_xyz.shape[0] == 0:
        return 0
    
    # 投影3D点到图像平面
    world_view_transform = viewpoint_cam.world_view_transform
    pts_in_cam = points_xyz @ world_view_transform[:3,:3] + world_view_transform[3,:3]
    
    # 只处理相机前方的点
    front_mask = pts_in_cam[:,2] > 0.1
    
    # 投影到像素坐标
    pts_proj_x = pts_in_cam[:,0] * viewpoint_cam.Fx / (pts_in_cam[:,2] + 1e-8) + viewpoint_cam.Cx
    pts_proj_y = pts_in_cam[:,1] * viewpoint_cam.Fy / (pts_in_cam[:,2] + 1e-8) + viewpoint_cam.Cy
    
    # 在图像范围内的点
    valid_mask = front_mask & \
                 (pts_proj_x >= 0) & (pts_proj_x < W) & \
                 (pts_proj_y >= 0) & (pts_proj_y < H)
    
    if valid_mask.sum() == 0:
        return 0
    
    # Step 5: 标记需要剪枝的点
    prune_mask = torch.zeros(points_xyz.shape[0], dtype=torch.bool, device=points_xyz.device)
    
    valid_indices = torch.where(valid_mask)[0]
    pts_x = pts_proj_x[valid_mask].long()
    pts_y = pts_proj_y[valid_mask].long()
    
    # 批量查询mask
    pts_x = torch.clamp(pts_x, 0, W-1)
    pts_y = torch.clamp(pts_y, 0, H-1)
    is_water_fog = water_fog_mask[pts_y, pts_x]
    
    prune_mask[valid_indices] = is_water_fog
    
    # Step 6: 额外条件：只剪低不透明度的点（更安全）
    dro_prune_opacity_threshold = getattr(opt, 'dro_prune_opacity_threshold', 0.5)
    opacity = gaussians.get_opacity.squeeze()
    if opacity.shape[0] == prune_mask.shape[0]:
        opacity_mask = opacity < dro_prune_opacity_threshold
        prune_mask = prune_mask & opacity_mask
    
    # Step 7: 执行剪枝
    pruned_count = prune_mask.sum().item()
    if pruned_count > 0:
        # 限制单次剪枝数量，避免过于激进
        max_prune_per_iter = getattr(opt, 'max_dro_prune_per_iter', 1000)
        if pruned_count > max_prune_per_iter:
            # 随机选择一部分剪枝
            prune_indices = torch.where(prune_mask)[0]
            perm = torch.randperm(pruned_count, device=prune_indices.device)
            selected_indices = prune_indices[perm[:max_prune_per_iter]]
            prune_mask = torch.zeros_like(prune_mask)
            prune_mask[selected_indices] = True
            pruned_count = max_prune_per_iter
        
        gaussians.prune_points(prune_mask)

        # 记录到 TensorBoard，方便观察 DRO 剪枝行为
        if (tb_writer is not None) and (iteration % 100 == 0):
            try:
                tb_writer.add_scalar('pruning/dro_pruned_points', pruned_count, iteration)
                tb_writer.add_scalar('pruning/dro_water_fog_pixels', water_fog_mask.sum().item(), iteration)
                tb_writer.add_scalar('pruning/total_points_after', gaussians.get_xyz.shape[0], iteration)
            except Exception as e:
                # 避免因为监控出错影响训练
                print(f"[ITER {iteration}] Warning: failed to log DRO pruning to TensorBoard: {e}")
        
        if iteration % 500 == 0:
            print(f"[ITER {iteration}] DRO-guided pruning: removed {pruned_count} water-fog points "
                  f"(water_fog_pixels: {water_fog_mask.sum().item()}, total_points: {points_xyz.shape[0]})")
    
    return pruned_count

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # LPIPS (VGG) for fake-air supervision (exp01 behavior). Compute sparsely for speed.
    lpips_net = None
    if float(getattr(opt, "lambda_fake_air_lpips", 0.0)) > 0.0:
        lpips_net = LPIPS(net_type="vgg").cuda().eval()
        for p in lpips_net.parameters():
            p.requires_grad_(False)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    scene_cameras.set_metric_normals_loading(opt.use_metric_normal_prior)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    # Complete underwater physics model
    complete_physics_model = None
    medium_model = None  # Keep for backward compatibility
    
    if pipe.use_underwater_model:
        # Use complete physics model instead of simple medium estimator
        # Initialize with custom underwater color: RGB(0, 105, 160)
        complete_physics_model = CompleteUnderwaterPhysics(
            appearance_dim=16,
            underwater_color=(0, 105, 160),
            max_depth=getattr(pipe, "physics_net_max_depth", 100.0),
        ).cuda()
        # 使用可配置的学习率重新创建优化器
        complete_physics_model.optimizer = torch.optim.Adam(
            complete_physics_model.parameters(), 
            lr=pipe.physics_model_lr, 
            betas=(0.9, 0.99)
        )
        # Keep original medium model as fallback
        medium_model = MediumEstimator(num_basis=pipe.medium_num_basis, use_spatial_smooth=True, underwater_color=(0, 105, 160)).cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    metric_normal_loss = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # IMPORTANT (3D-UIR style disentanglement):
        # If we use underwater/physics model, we must render plane_depth from the very beginning,
        # otherwise early 3DGS will bake water effects into Gaussians and disentanglement becomes hard.
        if getattr(pipe, "use_underwater_model", False):
            use_plane_depth = True
        else:
            # Backward-compatible gating when physics is disabled
            use_plane_depth = (iteration > opt.single_view_weight_from_iter) or \
                              (iteration > getattr(opt, 'depth_loss_from_iter', 500))

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model, 
                           medium_model=medium_model, complete_physics_model=complete_physics_model,
                           return_plane=use_plane_depth, return_depth_normal=iteration>opt.single_view_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # air (dehazed) output for logging and evaluation
        air_image = render_pkg.get("air_image", image)
        
        # Standard loss computation (like 3D-UIR)
        ssim_loss = (1.0 - ssim(image, gt_image))
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()

        # -------------------- Innovation 3: Background consistency loss (late-stage) --------------------
        lambda_bg = float(getattr(opt, "lambda_bg_consistency", 0.0))
        bg_from_iter = int(getattr(opt, "bg_loss_from_iter", 15000))
        bg_ramp = int(getattr(opt, "bg_loss_ramp_iters", 3000))
        bg_ramp = max(bg_ramp, 1)
        if lambda_bg > 0.0 and iteration >= bg_from_iter and ("rendered_alpha" in render_pkg) and ("plane_depth" in render_pkg):
            t = float(iteration - bg_from_iter) / float(bg_ramp)
            t = max(0.0, min(1.0, t))
            w_bg = lambda_bg * t

            alpha = render_pkg["rendered_alpha"].detach()
            depth2d = render_pkg.get("physics_depth", render_pkg["plane_depth"])
            depth2d = depth2d.squeeze(0) if depth2d.dim() == 3 else depth2d

            # Build bg mask & background color from GT only (robust gating).
            gt_gray_1hw = gt_image_gray if gt_image_gray.dim() == 3 else gt_image_gray.unsqueeze(0)
            hit_hw = None
            if bool(getattr(opt, "bg_use_hit_map", True)):
                # Prefer true pixel-wise hit map returned by renderer: hit_map = (plane_depth > 0).
                if "hit_map" in render_pkg and (render_pkg["hit_map"] is not None):
                    hit_hw = render_pkg["hit_map"].detach()
                else:
                    hit_hw = (depth2d > 0).float()
            mask_bg, B_gt, st = build_bg_mask_and_color(
                gt_image, gt_gray_1hw, depth2d, alpha, hit_hw=hit_hw,
                border=int(getattr(opt, "bg_border", 24)),
                grad_thr=float(getattr(opt, "bg_grad_thr", 0.02)),
                z_q=float(getattr(opt, "bg_z_q", 0.90)),
                z_tau_ratio=float(getattr(opt, "bg_z_tau_ratio", 0.05)),
                seed_top_q=float(getattr(opt, "bg_seed_top_q", 0.98)),
                min_seed_ratio=float(getattr(opt, "bg_min_seed_ratio", 0.01)),
                morph_k=int(getattr(opt, "bg_morph_k", 5)),
                seed_std_thr=float(getattr(opt, "bg_seed_std_thr", 0.05)),
                sigma_col_min=float(getattr(opt, "bg_sigma_col_min", 0.02)),
                sigma_col_max=float(getattr(opt, "bg_sigma_col_max", 0.12)),
            )

            min_pix = int(getattr(opt, "bg_min_pixels", 2048))
            if (mask_bg is not None) and (mask_bg.sum().item() >= max(min_pix, 10)):
                pred = image.clamp(0.0, 1.0)  # underwater render

                # Target color should come from physics model (recommended): B_inf or B(z).
                mode = str(getattr(opt, "bg_target_mode", "binf")).lower()
                target = None
                if mode == "bz":
                    if "backscatter" in render_pkg and (render_pkg["backscatter"] is not None):
                        target = render_pkg["backscatter"].detach().clamp(0.0, 1.0)  # [3,H,W]
                    elif "B_map" in render_pkg and (render_pkg["B_map"] is not None):
                        target = render_pkg["B_map"].detach().clamp(0.0, 1.0)
                if target is None and mode in ("binf", "bz"):
                    # Prefer B_inf if physics model exists (renderer already has it)
                    if complete_physics_model is not None and hasattr(complete_physics_model, "backscatter_net") and hasattr(complete_physics_model.backscatter_net, "B_inf"):
                        target = complete_physics_model.backscatter_net.B_inf.detach().clamp(0.0, 1.0)  # [3,1,1]
                if target is None:
                    # Fallback to GT-estimated constant background color (legacy)
                    target = B_gt.view(3, 1, 1).detach().clamp(0.0, 1.0)

                # Optional weak calibration: pull physics target towards B_gt when seed is very clean
                calib_w = float(getattr(opt, "bg_calib_weight", 0.0))
                if calib_w > 0.0 and target.dim() == 3 and target.shape[-2:] == pred.shape[-2:]:
                    # Only meaningful when target is a map; keep very weak.
                    target = (1.0 - calib_w) * target + calib_w * B_gt.view(3, 1, 1)

                # Soft-mask weighted L1
                if target.dim() == 3 and target.shape[-2:] == pred.shape[-2:]:
                    diff = (pred - target).abs() * mask_bg
                else:
                    diff = (pred - target).abs() * mask_bg
                denom = mask_bg.sum() * 3.0 + 1e-6
                L_bg_col = diff.sum() / denom

                # -------------------- Innovation 3 v2: split into two sub-tasks --------------------
                # (C1) bg_nohit_color: supervise background COLOR only on no-hit pixels (stable, avoids halos).
                # (C2) bg_hit_alpha: suppress far&weak alpha even in "hit" pixels inside background-like regions.
                #
                # C1 keeps legacy behavior.
                w_nohit_alpha = float(getattr(opt, "bg_alpha_weight", 0.2))
                a2d = alpha.squeeze(0) if alpha.dim() == 3 else alpha
                L_bg_nohit_alpha = ((a2d.clamp(0, 1) * mask_bg.squeeze(0)).sum() / (mask_bg.sum() + 1e-6))
                loss = loss + w_bg * (L_bg_col + w_nohit_alpha * L_bg_nohit_alpha)

                # C2 (optional, default OFF): bg_hit alpha suppression using GT+alpha+B_inf bg mask (aligned with pruning),
                # and sharing Scheme4 "far&weak" protection (p_alpha).
                bg_hit_w = float(getattr(opt, "bg_hit_alpha_weight", 0.0))
                bg_hit_from = int(getattr(opt, "bg_hit_from_iter", bg_from_iter))
                bg_hit_ramp = int(getattr(opt, "bg_hit_ramp_iters", bg_ramp))
                if bg_hit_w > 0.0 and iteration >= bg_hit_from:
                    t2 = float(iteration - bg_hit_from) / float(max(bg_hit_ramp, 1))
                    t2 = max(0.0, min(1.0, t2))
                    w2 = bg_hit_w * t2
                    try:
                        alpha2d = render_pkg["rendered_alpha"].squeeze(0).clamp(0.0, 1.0)
                        # Background-like mask that can overlap hits (do NOT pass nohit_hw)
                        binf_3 = None
                        if complete_physics_model is not None and hasattr(complete_physics_model, "backscatter_net") and hasattr(complete_physics_model.backscatter_net, "B_inf"):
                            binf_3 = complete_physics_model.backscatter_net.B_inf.detach().view(3)
                        bg_soft_full, _, st2 = build_bg_mask_gt_alpha_binf(
                            gt_image, gt_gray_1hw, alpha2d,
                            binf_3=binf_3,
                            nohit_hw=None,
                            alpha_bg_thr=float(getattr(opt, "bgm_alpha_bg_thr", 0.10)),
                            alpha_gamma=float(getattr(opt, "bgm_alpha_gamma", 1.5)),
                            gt_w=float(getattr(opt, "bgm_gt_w", 0.58)),
                            seed_gray_min=float(getattr(opt, "bgm_seed_gray_min", 0.03)),
                            seed_alpha_thr=float(getattr(opt, "bgm_seed_alpha_thr", 0.03)),
                            seed_binf_eps=float(getattr(opt, "bgm_seed_binf_eps", 0.35)),
                            min_seed_px=int(getattr(opt, "bgm_min_seed_px", 512)),
                            seed_std_thr=float(getattr(opt, "bgm_seed_std_thr", 0.08)),
                            binf_calib_w=float(getattr(opt, "bgm_binf_calib_w", 0.5)),
                            binf_eps=float(getattr(opt, "bgm_binf_eps", 0.25)),
                            use_binf_gate=bool(int(getattr(opt, "bgm_use_binf_gate", 1)) == 1),
                            sigma_col_min=float(getattr(opt, "bgm_sigma_col_min", 0.02)),
                            sigma_col_max=float(getattr(opt, "bgm_sigma_col_max", 0.15)),
                            col_thr=float(getattr(opt, "bgm_col_thr", 0.35)),
                            morph_k=int(getattr(opt, "bgm_morph_k", 5)),
                        )
                        if bg_soft_full is not None and (st2 is None or int(st2.get("skip", 0)) == 0):
                            # Far weight consistent with Scheme4
                            z_cut = float(getattr(pipe, "bg_z_cut", 60.0))
                            try:
                                if "scheme4_z_cut" in render_pkg and render_pkg["scheme4_z_cut"] is not None:
                                    zc = render_pkg["scheme4_z_cut"]
                                    z_cut = float(zc.item() if torch.is_tensor(zc) else zc)
                            except Exception:
                                pass
                            z_tau = float(getattr(pipe, "bg_z_tau", 5.0))
                            z_tau = max(z_tau, 1e-6)
                            depth_w = depth2d.detach()
                            far_w = torch.sigmoid((depth_w - z_cut) / z_tau).clamp(0.0, 1.0)

                            # Presence gate: only where there is actual Gaussian coverage
                            presence_thr = float(getattr(opt, "bg_prune_presence_thr", 1e-4))
                            presence = (alpha2d > presence_thr).float()

                            # Scheme4 far-protect p_alpha (far&weak gets penalized, far&strong protected)
                            if bool(getattr(pipe, "scheme4_far_protect", 1)):
                                a_prot = float(getattr(pipe, "scheme4_alpha_prot", 0.28))
                                a_tau = float(getattr(pipe, "scheme4_alpha_tau", 0.08))
                                a_tau = max(a_tau, 1e-6)
                                p_alpha = torch.sigmoid((alpha2d - a_prot) / a_tau).clamp(0.0, 1.0)
                                weak = (1.0 - p_alpha)
                            else:
                                weak = 1.0

                            # Soft mask for hit-alpha suppression
                            m_hit = (bg_soft_full.squeeze(0) * far_w * presence).clamp(0.0, 1.0)
                            denom2 = (m_hit.sum() + 1e-6)
                            L_bg_hit_alpha = (alpha2d * m_hit * weak).sum() / denom2
                            loss = loss + w2 * L_bg_hit_alpha
                            if tb_writer and (iteration % 10 == 0):
                                tb_writer.add_scalar("losses/bg_hit_alpha", float(L_bg_hit_alpha.detach().item()), iteration)
                    except Exception:
                        pass

                if tb_writer and (iteration % 10 == 0):
                    tb_writer.add_scalar("losses/bg_consistency_col", float(L_bg_col.detach().item()), iteration)
                    tb_writer.add_scalar("losses/bg_consistency_nohit_alpha", float(L_bg_nohit_alpha.detach().item()), iteration)
                    tb_writer.add_scalar("bg/mask_mean", float(st.get("mask_mean", 0.0)), iteration)
                    tb_writer.add_scalar("bg/seed_std", float(st.get("seed_std", 0.0)), iteration)
                    tb_writer.add_scalar("bg/seed_pixels", float(st.get("seed_pixels", 0.0)), iteration)
            else:
                if tb_writer and (iteration % 50 == 0):
                    tb_writer.add_scalar("bg/skip_view", 1.0, iteration)

        # --- Scheme 5: far-alpha regularization (optional, default OFF) ---
        # Encourage far regions (z >> z_cut) to have alpha -> 0 so residual far Gaussians naturally vanish.
        # We detach depth by default to avoid pushing geometry; only penalize opacity/alpha.
        lambda_far_alpha_cfg = float(getattr(pipe, "lambda_far_alpha", 0.0))
        # Optional ramp: make Scheme5 gentle early, stronger later (prevents early geometry suppression).
        if lambda_far_alpha_cfg > 0.0:
            far_from = int(getattr(pipe, "far_alpha_from_iter", 0))
            far_ramp = int(getattr(pipe, "far_alpha_ramp_iters", 0))
            if far_ramp > 0 and iteration >= far_from:
                t = float(iteration - far_from) / float(max(far_ramp, 1))
                t = max(0.0, min(1.0, t))
                lambda_far_alpha = lambda_far_alpha_cfg * t
            else:
                lambda_far_alpha = lambda_far_alpha_cfg
        else:
            lambda_far_alpha = 0.0

        if lambda_far_alpha > 0.0 and ("rendered_alpha" in render_pkg) and ("plane_depth" in render_pkg):
            alpha2d = render_pkg["rendered_alpha"].squeeze(0)  # (H,W)
            depth2d = render_pkg["plane_depth"]
            if depth2d.dim() == 3:
                depth2d = depth2d.squeeze(0)
            # Use the same effective z_cut as Scheme 4 (support auto-quantile z_cut if enabled).
            z_cut = float(getattr(pipe, "bg_z_cut", 60.0))
            # Prefer z_cut actually used by Scheme4 in the renderer (when available).
            try:
                if "scheme4_z_cut" in render_pkg and render_pkg["scheme4_z_cut"] is not None:
                    zc = render_pkg["scheme4_z_cut"]
                    z_cut = float(zc.item() if torch.is_tensor(zc) else zc)
            except Exception:
                pass
            if bool(getattr(pipe, "bg_z_cut_auto", 0)):
                try:
                    with torch.no_grad():
                        q = float(getattr(pipe, "bg_z_cut_q", 0.95))
                        q = max(0.5, min(0.9999, q))
                        z_min = float(getattr(pipe, "bg_z_cut_min", 10.0))
                        z_max = float(getattr(pipe, "bg_z_cut_max", 200.0))
                        z_fb = float(getattr(pipe, "bg_z_cut_fallback", z_cut))
                        hit_thr = float(getattr(pipe, "physics_alpha_thr", 1e-4))
                        valid = (depth2d > 0) & (alpha2d.detach().clamp(0, 1) > hit_thr)
                        if valid.any():
                            zq = float(torch.quantile(depth2d[valid].detach().float(), q).item())
                            z_cut = max(z_min, min(z_max, zq))
                        else:
                            z_cut = z_fb
                except Exception:
                    pass
            z_tau = float(getattr(pipe, "bg_z_tau", 5.0))
            z_tau = max(z_tau, 1e-6)
            if getattr(pipe, "far_alpha_detach_depth", True):
                depth_w = depth2d.detach()
            else:
                depth_w = depth2d
            far_w = torch.sigmoid((depth_w - z_cut) / z_tau)  # far->1
            # Scheme5: optional "far&weak" protection (decoupled from Scheme4).
            # Only penalize far alpha that is also "weak hit" (avoid suppressing far real geometry).
            # Backward compatible: if scheme5_far_protect is missing, fall back to scheme4_far_protect.
            use_s5_protect = getattr(pipe, "scheme5_far_protect", None)
            if use_s5_protect is None:
                use_s5_protect = getattr(pipe, "scheme4_far_protect", 1)
            if bool(use_s5_protect):
                a_prot = float(getattr(pipe, "scheme4_alpha_prot", 0.28))
                a_tau = float(getattr(pipe, "scheme4_alpha_tau", 0.08))
                a_tau = max(a_tau, 1e-6)
                p_alpha = torch.sigmoid((alpha2d.clamp(0, 1) - a_prot) / a_tau).clamp(0.0, 1.0)
                loss_far_alpha = (alpha2d * far_w * (1.0 - p_alpha)).mean()
            else:
                loss_far_alpha = (alpha2d * far_w).mean()
            loss = loss + lambda_far_alpha * loss_far_alpha
            if tb_writer and (iteration % 10 == 0):
                tb_writer.add_scalar('losses/far_alpha', float(loss_far_alpha.detach().item()), iteration)

        # --- NEW: Pseudo clear-scene supervision (fake air) on pure 3DGS render ---
        # Keep the proven-useful parts: masked L1 + multi-scale Laplacian (confidence-weighted).
        fake_air_gt = None
        fake_air_gray = None
        fake_air_w = None  # [1,H,W] weight (mask * confidence)

        fake_air_loss_val = 0.0   # masked L1 (for progress bar / legacy logging)
        fake_air_lap_val = 0.0
        fake_air_lpips_val = 0.0

        lambda_fake_air = float(getattr(opt, "lambda_fake_air", 0.0))
        lambda_fake_air_lap = float(getattr(opt, "lambda_fake_air_lap", 0.0))
        lambda_fake_air_lpips = float(getattr(opt, "lambda_fake_air_lpips", 0.0))
        fake_air_lpips_interval = int(getattr(opt, "fake_air_lpips_interval", 20))
        fake_air_lpips_downsample = int(getattr(opt, "fake_air_lpips_downsample", 2))

        if (lambda_fake_air > 0.0 or lambda_fake_air_lap > 0.0 or lambda_fake_air_lpips > 0.0) and \
           iteration >= getattr(opt, "fake_air_loss_from_iter", 0):
            fake_air_gt, fake_air_gray = viewpoint_cam.get_fake_air_image() if hasattr(viewpoint_cam, "get_fake_air_image") else (None, None)
            if fake_air_gt is not None and ("air_image" in render_pkg):
                # --- Optional: gate air_image to black with the SAME gating logic as Scheme 4 ---
                # Motivation: make fake-air supervision consistent with "far treated as background".
                # Default OFF via PipelineParams.train_air_bg_black.
                try:
                    if bool(getattr(pipe, "train_air_bg_black", 0)) and ("rendered_alpha" in render_pkg) and ("plane_depth" in render_pkg):
                        alpha2d = render_pkg["rendered_alpha"].squeeze(0).clamp(0.0, 1.0)  # (H,W)
                        depth2d = render_pkg["plane_depth"]
                        if depth2d.dim() == 3:
                            depth2d = depth2d.squeeze(0)

                        alpha_thr = float(getattr(pipe, "train_air_bg_alpha_thr", -1.0))
                        if alpha_thr <= 0:
                            alpha_thr = float(getattr(pipe, "physics_alpha_thr", 1e-4))
                        hit = (alpha2d > alpha_thr)
                        a_hit = torch.where(hit, alpha2d, torch.zeros_like(alpha2d))

                        z_cut = float(getattr(pipe, "train_air_bg_z_cut", -1.0))
                        if z_cut <= 0:
                            z_cut = float(getattr(pipe, "bg_z_cut", 60.0))
                            # Optional: auto-quantile z_cut (per-view) consistent with Scheme 4.
                            if bool(getattr(pipe, "bg_z_cut_auto", 0)):
                                try:
                                    with torch.no_grad():
                                        q = float(getattr(pipe, "bg_z_cut_q", 0.95))
                                        q = max(0.5, min(0.9999, q))
                                        z_min = float(getattr(pipe, "bg_z_cut_min", 10.0))
                                        z_max = float(getattr(pipe, "bg_z_cut_max", 200.0))
                                        z_fb = float(getattr(pipe, "bg_z_cut_fallback", z_cut))
                                        valid = (depth2d > 0) & (alpha2d > alpha_thr)
                                        if valid.any():
                                            zq = float(torch.quantile(depth2d[valid].detach().float(), q).item())
                                            z_cut = max(z_min, min(z_max, zq))
                                        else:
                                            z_cut = z_fb
                                except Exception:
                                    pass
                        z_tau = float(getattr(pipe, "train_air_bg_z_tau", -1.0))
                        if z_tau <= 0:
                            z_tau = float(getattr(pipe, "bg_z_tau", 5.0))
                        z_tau = max(z_tau, 1e-6)

                        if bool(getattr(pipe, "physics_depth_gated_background", 0)):
                            far_w = torch.sigmoid((depth2d - z_cut) / z_tau)  # far->1
                            # Optional: protect far-but-strong geometry from being blacked out.
                            if bool(getattr(pipe, "scheme4_far_protect", 1)):
                                a_prot = float(getattr(pipe, "scheme4_alpha_prot", 0.28))
                                a_tau = float(getattr(pipe, "scheme4_alpha_tau", 0.08))
                                a_tau = max(a_tau, 1e-6)
                                p_alpha = torch.sigmoid((a_hit - a_prot) / a_tau).clamp(0.0, 1.0)
                                gate = torch.clamp(1.0 - far_w * (1.0 - p_alpha), 0.0, 1.0)
                                a_eff = torch.clamp(a_hit * gate, 0.0, 1.0)
                            else:
                                a_eff = torch.clamp(a_hit * (1.0 - far_w), 0.0, 1.0)
                        else:
                            a_eff = a_hit

                        air_image = torch.clamp(air_image * a_eff.unsqueeze(0), 0.0, 1.0)
                except Exception:
                    pass

                # Prefer alpha mask if available (more robust than plane_depth > 0)
                if "rendered_alpha" in render_pkg:
                    m = render_pkg["rendered_alpha"].detach()  # [1,H,W]
                    mask = (m > 1e-4).float()
                else:
                    pd = _safe_2d(render_pkg.get("plane_depth", None))
                    mask = (pd > 1e-6).float().unsqueeze(0) if pd is not None else \
                           torch.ones((1, air_image.shape[1], air_image.shape[2]), device=air_image.device, dtype=air_image.dtype)

                # --- Heuristic confidence (no uncertainty map available) ---
                # Down-weight regions where fake-air has much stronger edges than underwater GT (likely hallucinations),
                # and down-weight far-depth pixels (often water-dominated, pseudo GT less reliable).
                conf = torch.ones_like(mask)
                try:
                    if fake_air_gray is None:
                        fake_air_gray = (0.299 * fake_air_gt[0] + 0.587 * fake_air_gt[1] + 0.114 * fake_air_gt[2]).clamp(0, 1).unsqueeze(0)
                    g_fake = _sobel_grad_mag(fake_air_gray)
                    g_uwa = _sobel_grad_mag(gt_image_gray)
                    ratio = g_fake / (g_uwa + 1e-3)
                    r_thr = float(getattr(opt, "fake_air_edge_ratio_thr", 2.0))
                    k = float(getattr(opt, "fake_air_edge_ratio_k", 1.0))
                    conf_edge = torch.exp(-k * torch.relu(ratio - r_thr))
                    conf = conf * conf_edge
                except Exception:
                    pass

                depth_z = _safe_2d(render_pkg.get("physics_depth", render_pkg.get("plane_depth", None)))
                if depth_z is not None:
                    tau = float(getattr(opt, "fake_air_depth_tau", 12.0))
                    conf_depth = torch.exp(-torch.clamp(depth_z, 0.0, 100.0) / max(tau, 1e-6)).unsqueeze(0)
                    conf = conf * conf_depth

                conf = torch.clamp(conf, 0.0, 1.0).detach()
                fake_air_w = (mask * conf).detach()

                # (1) masked L1
                diff = torch.abs(air_image - fake_air_gt)
                denom = (fake_air_w.sum() * 3.0) + 1e-6
                fake_air_l1 = (diff * fake_air_w).sum() / denom
                loss = loss + lambda_fake_air * fake_air_l1
                fake_air_loss_val = float(fake_air_l1.detach().item())

                # (2) multi-scale Laplacian (sharpness / high-freq)
                if lambda_fake_air_lap > 0.0:
                    lap = _multiscale_lap_loss(air_image, fake_air_gt, fake_air_w, scales=(1, 2))
                    loss = loss + lambda_fake_air_lap * lap
                    fake_air_lap_val = float(lap.detach().item())

                # (3) LPIPS (VGG) - computed sparsely for speed (exp01 behavior)
                if (lambda_fake_air_lpips > 0.0) and (lpips_net is not None) and (iteration % max(fake_air_lpips_interval, 1) == 0):
                    ds = max(fake_air_lpips_downsample, 1)
                    a = air_image.unsqueeze(0)
                    b = fake_air_gt.unsqueeze(0)
                    wm = fake_air_w.unsqueeze(0)
                    if ds > 1:
                        a = F.interpolate(a, scale_factor=1.0 / ds, mode="bilinear", align_corners=False)
                        b = F.interpolate(b, scale_factor=1.0 / ds, mode="bilinear", align_corners=False)
                        wm = F.interpolate(wm, scale_factor=1.0 / ds, mode="nearest")
                    # Fill invalid pixels with 0.5 to avoid background dominating LPIPS
                    a = a * wm + 0.5 * (1.0 - wm)
                    b = b * wm + 0.5 * (1.0 - wm)
                    lp = lpips_net(torch.clamp(a, 0.0, 1.0), torch.clamp(b, 0.0, 1.0)).mean()
                    loss = loss + lambda_fake_air_lpips * lp
                    fake_air_lpips_val = float(lp.detach().item())
        
        # --- 混合方案C: DRO剪枝代替DRO Loss ---
        # 关闭DRO Loss，使用DRO指导剪枝来消除水体雾墙
        Ll1depth_val = 0.0
        
        # 检查是否使用DRO Loss（默认关闭）
        use_dro_loss = getattr(opt, 'use_dro_loss', False)
        
        if use_dro_loss:
            # 如果启用DRO Loss，保留原有逻辑（兼容性）
            depth_loss_start_iter = getattr(opt, 'depth_loss_from_iter', 500)
            
            if iteration > depth_loss_start_iter and 'plane_depth' in render_pkg and viewpoint_cam.depth_reliable:
                plane_distance = render_pkg['plane_depth']
                if plane_distance.dim() == 2:
                    plane_distance = plane_distance.unsqueeze(0)
                
                pred_inv_depth = 1.0 / (plane_distance + 1e-5)
                gt_inv_depth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
                
                pred_masked = pred_inv_depth * depth_mask
                gt_masked = gt_inv_depth * depth_mask
                
                pred_p5 = torch.quantile(pred_masked[depth_mask > 0.5], 0.05)
                pred_p95 = torch.quantile(pred_masked[depth_mask > 0.5], 0.95)
                gt_p5 = torch.quantile(gt_masked[depth_mask > 0.5], 0.05)
                gt_p95 = torch.quantile(gt_masked[depth_mask > 0.5], 0.95)
                
                pred_inv_depth_norm = torch.clamp((pred_inv_depth - pred_p5) / (pred_p95 - pred_p5 + 1e-8), 0, 1)
                gt_inv_depth_norm = torch.clamp((gt_inv_depth - gt_p5) / (gt_p95 - gt_p5 + 1e-8), 0, 1)
                
                Ll1depth = torch.abs((pred_inv_depth_norm - gt_inv_depth_norm) * depth_mask).mean()
                lambda_depth = getattr(opt, 'lambda_depth', 0.05)
                loss += lambda_depth * Ll1depth
                Ll1depth_val = Ll1depth.item()
                
                lambda_smooth = getattr(opt, 'lambda_smooth', 0.01)
                if lambda_smooth > 0:
                    Lsmooth = lambda_smooth * depth_aware_smooth_loss(pred_inv_depth_norm, image, depth_mask)
                    loss += Lsmooth
                
                lambda_tv = getattr(opt, 'lambda_tv', 0.1)
                tv_from_iter = getattr(opt, 'tv_from_iter', 500)
                tv_until_iter = getattr(opt, 'tv_until_iter', 30000)
                if lambda_tv > 0 and iteration > tv_from_iter and iteration < tv_until_iter:
                    tv = anisotropic_total_variation_loss(pred_inv_depth_norm.squeeze())
                    loss += lambda_tv * tv
                
                if iteration % 10 == 0 and tb_writer:
                    tb_writer.add_scalar('losses/depth_loss', Ll1depth.item(), iteration)
        else:
            # 混合方案C: 使用DRO剪枝而非Loss
            # DRO剪枝会在densification阶段调用（见下方densify_and_prune部分）
            pass

        # 初始化各项损失（用于监控）
        L_tv = torch.tensor(0.0, device=loss.device)
        L_physics = torch.tensor(0.0, device=loss.device)
        L_scale = torch.tensor(0.0, device=loss.device)
        current_physics_constraint_weight = 0.0  # 动态物理约束权重（用于监控，建议5）

        # Complete physics regularization
        if complete_physics_model is not None and ("backscatter" in render_pkg and "attenuation" in render_pkg):
            backscatter = render_pkg["backscatter"]
            attenuation = render_pkg["attenuation"]
            # Use the exact metric depth used by the renderer physics (z in meters)
            depth_map = render_pkg.get("physics_depth", render_pkg["plane_depth"])
            if depth_map.dim() == 3:
                depth_map = depth_map.squeeze(0)  # (1, H, W) -> (H, W)
            
            # 数值稳定性检查
            if torch.isnan(backscatter).any() or torch.isnan(attenuation).any():
                print(f"[ITER {iteration}] Warning: NaN detected in physics components, skipping physics loss")
                backscatter = torch.zeros_like(backscatter)
                attenuation = torch.ones_like(attenuation)
            
            if torch.isnan(depth_map).any() or (depth_map == 0).all():
                print(f"[ITER {iteration}] Warning: Invalid depth map, skipping physics loss")
                depth_map = torch.ones_like(depth_map) * 0.1  # 使用默认深度
            
            # Debug: Print physics components statistics
            if iteration % 1000 == 0:
                print(f"[ITER {iteration}] Physics Debug:")
                print(f"  Backscatter: mean={backscatter.mean():.4f}, std={backscatter.std():.4f}, range=[{backscatter.min():.4f}, {backscatter.max():.4f}]")
                print(f"  Attenuation: mean={attenuation.mean():.4f}, std={attenuation.std():.4f}, range=[{attenuation.min():.4f}, {attenuation.max():.4f}]")
                print(f"  Depth: mean={depth_map.mean():.4f}, std={depth_map.std():.4f}, range=[{depth_map.min():.4f}, {depth_map.max():.4f}]")
            
            # TV regularization for physics components (使用可配置的TV权重)
            tv_w = getattr(pipe, 'physics_tv_weight', pipe.medium_tv_weight if hasattr(pipe, 'medium_tv_weight') else 0.005)
            L_tv = tv_w * (CompleteUnderwaterPhysics.tv_loss(backscatter) + 
                           CompleteUnderwaterPhysics.tv_loss(attenuation))
            
            # Physics constraints (使用可配置的约束权重，建议5: 动态调度)
            base_phys_w = getattr(pipe, 'physics_constraint_weight', 0.1)
            # 0-3000: 10% 基础权重；3000-10000: 线性从10%→100%；10000+：100%
            if iteration < 3000:
                phys_factor = 0.1
            elif iteration < 10000:
                alpha = float(iteration - 3000) / 7000.0
                alpha = max(0.0, min(1.0, alpha))
                phys_factor = 0.1 + 0.9 * alpha
            else:
                phys_factor = 1.0
            physics_constraint_weight = base_phys_w * phys_factor
            current_physics_constraint_weight = physics_constraint_weight
            L_physics = physics_constraint_weight * complete_physics_model.physics_constraints(backscatter, attenuation, depth_map)
            
            loss += L_tv + L_physics
            
        elif medium_model is not None and ("a_map" in render_pkg and "B_map" in render_pkg):
            # Fallback to original medium model
            a_map = render_pkg["a_map"].detach() if not render_pkg["a_map"].requires_grad else render_pkg["a_map"]
            B_map = render_pkg["B_map"].detach() if not render_pkg["B_map"].requires_grad else render_pkg["B_map"]
            
            tv_w = pipe.medium_tv_weight if hasattr(pipe, 'medium_tv_weight') else 0.01
            L_medium = tv_w * (MediumEstimator.tv_loss(a_map) + MediumEstimator.tv_loss(B_map))
            L_physics = physics_constraints(a_map, B_map, render_pkg)
            
            loss += L_medium + L_physics
        
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            L_scale = opt.scale_loss_weight * min_scale_loss.mean()
            loss += L_scale
        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += (normal_loss)

        # Metric3D normal prior (optional, weak regularization on geometry)
        if getattr(opt, "use_metric_normal_prior", False) and \
           iteration > getattr(opt, "metric_normal_from_iter", opt.single_view_weight_from_iter):

            if hasattr(viewpoint_cam, "metric_normal") and viewpoint_cam.metric_normal is not None:
                # Predicted normal from PGSR (camera/view space, [-1,1])
                pred_normal = render_pkg["rendered_normal"]          # [3,H,W]
                # Metric3D normal prior (already resized in Camera to match resolution)
                metric_normal = viewpoint_cam.metric_normal.to(pred_normal.device)  # [3,H,W]

                # Normalize both to be safe
                pred_normal_n = F.normalize(pred_normal, dim=0)
                metric_normal_n = F.normalize(metric_normal, dim=0)

                # Cosine similarity per pixel
                cos_sim = torch.clamp((pred_normal_n * metric_normal_n).sum(0), -1.0, 1.0)  # [H,W]
                ang_loss = 1.0 - cos_sim  # 0 when directions align, up to 2 when opposite

                # Confidence weighting (from Metric3D)
                metric_conf = getattr(viewpoint_cam, "metric_normal_conf", None)
                if metric_conf is not None:
                    conf = metric_conf.to(pred_normal.device)
                    # conf: [1,H,W] or [H,W]
                    if conf.dim() == 3 and conf.shape[0] == 1:
                        conf = conf[0]  # [H,W]
                    conf_min = getattr(opt, "metric_conf_min", 0.0)
                    conf_max = getattr(opt, "metric_conf_max", 8.0)
                    conf_clamped = torch.clamp(conf, conf_min, conf_max)
                    weight_map = (conf_clamped - conf_min) / (conf_max - conf_min + 1e-6)
                else:
                    weight_map = torch.ones_like(ang_loss)

                # Mask out very low-confidence regions
                mask = weight_map > 0.1
                if mask.any():
                    weighted_loss = (weight_map * ang_loss)[mask].mean()
                    lambda_metric = getattr(opt, "lambda_metric_normal", 0.02)
                    metric_normal_loss = lambda_metric * weighted_loss
                    loss += metric_normal_loss

        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            # 渐进式激活多视图权重：在 [multi_view_weight_from_iter, multi_view_weight_from_iter + rampup] 之间线性增长
            start_iter = opt.multi_view_weight_from_iter
            rampup_iters = getattr(opt, 'multi_view_weight_rampup_iter', 3000)
            end_iter = start_iter + rampup_iters
            if iteration < end_iter:
                alpha = float(iteration - start_iter) / float(rampup_iters)
                alpha = max(0.0, min(1.0, alpha))
            else:
                alpha = 1.0
            ncc_weight = opt.multi_view_ncc_weight * alpha
            geo_weight = opt.multi_view_geo_weight * alpha

            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ## compute geometry consistency mask and loss
                # Ensure plane_depth is 2D
                plane_depth_2d = render_pkg['plane_depth']
                if plane_depth_2d.dim() == 3:
                    plane_depth_2d = plane_depth_2d.squeeze(0)  # (1, H, W) -> (H, W)
                H, W = plane_depth_2d.shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(plane_depth_2d.device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                # get_points_from_depth accepts any shape (will squeeze internally)
                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                
                # get_points_depth_in_depth_map expects (1, H, W) - ensure correct shape
                nearest_plane_depth = nearest_render_pkg['plane_depth']
                if nearest_plane_depth.dim() == 2:
                    nearest_plane_depth = nearest_plane_depth.unsqueeze(0)  # (H, W) -> (1, H, W)
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_plane_depth, pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            # Use clear-scene (air) grayscale for LNCC to align multi-view to geometry/J, not underwater I.
                            air_gray = (0.299 * air_image[0] + 0.587 * air_image[1] + 0.114 * air_image[2]).clamp(0, 1).unsqueeze(0)  # (1,H,W)
                            H, W = air_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(air_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        nearest_air = nearest_render_pkg.get("air_image", nearest_render_pkg["render"])
                        nearest_air_gray = (0.299 * nearest_air[0] + 0.587 * nearest_air[1] + 0.114 * nearest_air[2]).clamp(0, 1).unsqueeze(0)  # (1,H,W)
                        sampled_gray_val = F.grid_sample(nearest_air_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        # 数值稳定性检查
        if torch.isnan(loss) or torch.isinf(loss):
            # IMPORTANT: Even if we skip the "real" backward, densification stats later rely on
            # viewspace_point_tensor.grad (and viewspace_points_abs.grad). If we simply skip backward,
            # those grads stay None and add_densification_stats will crash.
            #
            # Fix: backprop a safe zero loss that is connected to the viewspace tensors, producing
            # zero gradients (not None). This keeps training robust without introducing NaNs.
            print(f"[ITER {iteration}] Warning: Invalid loss detected (NaN or Inf), using safe zero-backward")
            safe = viewspace_point_tensor.sum() * 0.0
            v_abs = render_pkg.get("viewspace_points_abs", None)
            if v_abs is not None:
                safe = safe + v_abs.sum() * 0.0
            safe.backward()
            loss = safe.detach()
        else:
            loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar with detailed loss tracking
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            
            # 记录各项损失到TensorBoard（每10次迭代）
            if iteration % 10 == 0 and tb_writer:
                tb_writer.add_scalar('losses/image_loss', image_loss.item(), iteration)
                tb_writer.add_scalar('losses/l1_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('losses/ssim_loss', ssim_loss.item(), iteration)
                tb_writer.add_scalar('losses/tv_loss', L_tv.item(), iteration)
                tb_writer.add_scalar('losses/physics_constraint_loss', L_physics.item(), iteration)
                tb_writer.add_scalar('losses/scale_loss', L_scale.item(), iteration)
                if metric_normal_loss is not None:
                    tb_writer.add_scalar('losses/metric_normal_loss', metric_normal_loss.item(), iteration)
                if normal_loss is not None:
                    tb_writer.add_scalar('losses/normal_loss', normal_loss.item(), iteration)
                if geo_loss is not None:
                    tb_writer.add_scalar('losses/geo_loss', geo_loss.item(), iteration)
                if ncc_loss is not None:
                    tb_writer.add_scalar('losses/ncc_loss', ncc_loss.item(), iteration)
                tb_writer.add_scalar('losses/total_loss', loss.item(), iteration)

                # Optional fake-air supervision logging
                if float(getattr(opt, "lambda_fake_air", 0.0)) > 0.0:
                    tb_writer.add_scalar('losses/fake_air_l1', fake_air_loss_val, iteration)
                if float(getattr(opt, "lambda_fake_air_lap", 0.0)) > 0.0:
                    tb_writer.add_scalar('losses/fake_air_lap', fake_air_lap_val, iteration)
                if float(getattr(opt, "lambda_fake_air_lpips", 0.0)) > 0.0:
                    tb_writer.add_scalar('losses/fake_air_lpips', fake_air_lpips_val, iteration)

                # 记录超参数（每100次迭代）
                if iteration % 100 == 0:
                    tb_writer.add_scalar('hyperparams/physics_model_lr', pipe.physics_model_lr, iteration)
                    tb_writer.add_scalar('hyperparams/physics_tv_weight', getattr(pipe, 'physics_tv_weight', 0.005), iteration)
                    # 使用当前动态物理约束权重而不是静态配置（建议5）
                    tb_writer.add_scalar('hyperparams/physics_constraint_weight', current_physics_constraint_weight, iteration)
                    tb_writer.add_scalar('hyperparams/physics_region_depth_threshold', getattr(pipe, 'physics_region_depth_threshold', 2.0), iteration)
                    tb_writer.add_scalar('hyperparams/water_region_depth_threshold', getattr(pipe, 'water_region_depth_threshold', 1.5), iteration)
                    tb_writer.add_scalar('hyperparams/physics_region_gradient_scale', getattr(pipe, 'physics_region_gradient_scale', 0.1), iteration)
                    tb_writer.add_scalar('hyperparams/water_region_gradient_scale', getattr(pipe, 'water_region_gradient_scale', 0.05), iteration)

                    # 建议6：点云统计监控（每100次迭代）
                    try:
                        total_points = gaussians.get_xyz.shape[0]
                        tb_writer.add_scalar('points/total_points', total_points, iteration)
                        opacity_vals = gaussians.get_opacity
                        if opacity_vals.numel() > 0:
                            tb_writer.add_scalar('points/opacity_mean', opacity_vals.mean().item(), iteration)
                            tb_writer.add_scalar('points/opacity_median', opacity_vals.median().item(), iteration)
                        # 按深度区间统计点数
                        if total_points > 0:
                            pts_xyz_all = gaussians.get_xyz
                            pts_depth_all = torch.norm(pts_xyz_all, dim=1)
                            near_mask = pts_depth_all < 5.0
                            mid_mask = (pts_depth_all >= 5.0) & (pts_depth_all < 20.0)
                            far_mask = pts_depth_all >= 20.0
                            tb_writer.add_scalar('points/near_0_5m', near_mask.sum().item(), iteration)
                            tb_writer.add_scalar('points/mid_5_20m', mid_mask.sum().item(), iteration)
                            tb_writer.add_scalar('points/far_20m_plus', far_mask.sum().item(), iteration)
                    except Exception as e:
                        print(f"[ITER {iteration}] Warning: failed to log point statistics: {e}")
            
            if iteration % 10 == 0:
                # Keep tqdm postfix short for easier copy/paste (optional).
                # All detailed scalars remain in TensorBoard.
                if bool(getattr(opt, "tqdm_short", False)):
                    # Last bg prune count (if any) is logged to TB; for console, infer from current step's pruning.
                    # If pruning did not run this iter, show 0.
                    try:
                        bg_prune_num = int(getattr(gaussians, "_bg_prune_last_num", 0))
                    except Exception:
                        bg_prune_num = 0

                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "L1": f"{Ll1.item():.{5}f}",
                        "SSIM": f"{ssim_loss.item():.{5}f}",
                        "FakeAir": f"{fake_air_loss_val:.{5}f}",
                        "BGPrune": f"{bg_prune_num}",
                        "Points": f"{gaussians.get_xyz.shape[0]}",
                    }
                else:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "L1": f"{Ll1.item():.{5}f}",
                        "SSIM": f"{ssim_loss.item():.{5}f}",
                        "Depth": f"{Ll1depth_val:.{5}f}",
                        "TV": f"{L_tv.item():.{5}f}",
                        "Phys": f"{L_physics.item():.{5}f}",
                        "FakeAir": f"{fake_air_loss_val:.{5}f}",
                        "Single": f"{ema_single_view_for_log:.{5}f}",
                        "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                        "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                        "Points": f"{gaussians.get_xyz.shape[0]}",
                    }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background),
                app_model=app_model, medium_model=medium_model, complete_physics_model=complete_physics_model
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            # Densification（还原为原版 PGSR 的策略）
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # ---------- Innovation 1: block far densify (capacity routing) ----------
                # Exclude far Gaussians from densify stats so they are unlikely to clone/split in far background.
                visibility_filter_near = visibility_filter
                try:
                    if getattr(pipe, "densify_block_far", False):
                        start_iter = int(getattr(pipe, "densify_block_far_from_iter", 0))
                        if start_iter <= 0:
                            # 4.8.2: earlier default start for far-blocking (can still be overridden by CLI).
                            start_iter = int(opt.densify_from_iter) + 500
                        if iteration >= start_iter:
                            xyz = gaussians.get_xyz  # [N,3]
                            wv = viewpoint_cam.world_view_transform
                            pts_cam = xyz @ wv[:3, :3] + wv[3, :3]
                            z_cam = pts_cam[:, 2]

                            depth2d = render_pkg.get("plane_depth", None)
                            alpha2d = render_pkg.get("rendered_alpha", None)
                            z_cut_q = float(getattr(pipe, "densify_z_cut_q", 0.90))
                            z_cut_min = float(getattr(pipe, "densify_z_cut_min", 10.0))
                            z_cut_max = float(getattr(pipe, "densify_z_cut_max", 80.0))
                            z_cut_fallback = float(getattr(pipe, "densify_z_cut_fallback", 20.0))
                            hit_thr = float(getattr(pipe, "densify_hit_alpha_thr", 1e-4))

                            if depth2d is not None:
                                if depth2d.dim() == 3:
                                    depth2d = depth2d.squeeze(0)
                                if alpha2d is not None:
                                    if alpha2d.dim() == 3:
                                        alpha2d = alpha2d.squeeze(0)
                                    valid = (depth2d > 0) & (alpha2d > hit_thr)
                                else:
                                    valid = (depth2d > 0)
                                if valid.any():
                                    q = max(0.5, min(0.9999, z_cut_q))
                                    z_cut = float(torch.quantile(depth2d[valid].detach().float(), q).item())
                                    z_cut = max(z_cut_min, min(z_cut_max, z_cut))
                                else:
                                    z_cut = z_cut_fallback
                            else:
                                z_cut = z_cut_fallback

                            # Optional ramp-in: loose -> tight threshold after start_iter.
                            # Effective z_cut interpolates from z_cut_max (loose, keeps more points) to computed z_cut (tight).
                            ramp = int(getattr(pipe, "densify_block_far_ramp_iters", 0))
                            if ramp > 0:
                                t = float(iteration - start_iter) / float(max(ramp, 1))
                                t = max(0.0, min(1.0, t))
                                z_loose = float(z_cut_max)
                                z_cut = (1.0 - t) * z_loose + t * float(z_cut)

                            visibility_filter_near = visibility_filter & (z_cam < float(z_cut))
                except Exception as e:
                    # Keep training robust; fallback to original behavior on any unexpected shape/device issue.
                    visibility_filter_near = visibility_filter

                mask = (render_pkg["out_observe"] > 0) & visibility_filter_near
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                # Robustness: if for any reason grads are not populated, skip densify stats this iteration.
                if (viewspace_point_tensor.grad is None) or (viewspace_point_tensor_abs.grad is None):
                    if iteration % 100 == 0:
                        print(f"[ITER {iteration}] Warning: missing viewspace grads, skip add_densification_stats")
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter_near)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 使用与原版 PGSR 相同的 densify_and_prune 逻辑
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.densify_abs_grad_threshold,
                        opt.opacity_cull_threshold,
                        scene.cameras_extent,
                        size_threshold
                    )

                    # 额外：在标准 densify 之后执行一次“温和”的 DRO 引导剪枝
                    # （只在 opt.dro_guided_pruning=True 时启用）
                    if getattr(opt, 'dro_guided_pruning', False):
                        dro_guided_water_pruning(gaussians, render_pkg, viewpoint_cam, opt, iteration, tb_writer)

                    # 额外：背景“双证据触发”稳健剪枝（Scheme4 ∩ Innovation3）
                    _bg_double_evidence_prune(
                        gaussians, render_pkg, viewpoint_cam,
                        gt_image, (gt_image_gray if gt_image_gray.dim() == 3 else gt_image_gray.unsqueeze(0)),
                        pipe, opt, iteration,
                        tb_writer=tb_writer
                    )

            # --- BG-mask pruning can (and should) continue after densify_until_iter ---
            # Motivation: the far-blue "fog wall" Gaussians may persist / re-emerge after densification ends.
            # The pruning function already has its own gating (bg_prune_enable/from_iter/interval/until_iter),
            # so calling it here every iteration is safe and cheap.
            _bg_double_evidence_prune(
                gaussians, render_pkg, viewpoint_cam,
                gt_image, (gt_image_gray if gt_image_gray.dim() == 3 else gt_image_gray.unsqueeze(0)),
                pipe, opt, iteration,
                tb_writer=tb_writer
            )
            
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity (3D-UIR style aggressive reset)
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    # Use aggressive opacity reset if enabled (like 3D-UIR)
                    reset_value = getattr(opt, 'reset_opacity_value', 0.01) if getattr(opt, 'aggressive_opacity_reset', True) else 0.01
                    gaussians.reset_opacity(reset_value=reset_value)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                
                # Update complete physics model (like 3D-UIR)
                if complete_physics_model is not None:
                    complete_physics_model.optimizer.step()
                    complete_physics_model.optimizer.zero_grad(set_to_none=True)
                elif medium_model is not None:
                    medium_model.optimizer.step()
                    medium_model.optimizer.zero_grad(set_to_none=True)
                
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)
                
                if complete_physics_model is not None:
                    complete_physics_model.save_weights(scene.model_path, iteration)
                elif medium_model is not None:
                    medium_model.save_weights(scene.model_path, iteration)
    
    app_model.save_weights(scene.model_path, opt.iterations)
    if complete_physics_model is not None:
        complete_physics_model.save_weights(scene.model_path, opt.iterations)
    elif medium_model is not None:
        medium_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    # Note: cfg_args will be saved in main() before calling training() to ensure
    # all PipelineParams and OptimizationParams are included
    # 写一个 minimal 场景路径记录，便于评估脚本自动发现源数据路径
    scene_cfg = {
        "source_path": args.source_path,
        "air_source_path": getattr(args, 'air_source_path', ""),
        "fake_air_source_path": getattr(args, 'fake_air_source_path', ""),
        "fake_air_folder_name": getattr(args, 'fake_air_folder_name', "fake_air_Deep_Sea-NN")
    }
    with open(os.path.join(args.model_path, "scene_paths.json"), 'w') as f:
        import json
        json.dump(scene_cfg, f)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene : Scene, renderFunc, renderArgs,
                    app_model,
                    medium_model=None,
                    complete_physics_model=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # IMPORTANT: evaluation should use the same underwater rendering path as training
                    out = renderFunc(
                        viewpoint, scene.gaussians, *renderArgs,
                        app_model=app_model, medium_model=medium_model, complete_physics_model=complete_physics_model,
                        return_plane=True, return_depth_normal=False
                    )
                    # Underwater rendered image (I = J*A + B when physics enabled)
                    image = out["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # Also log the clear-scene (air) image for debugging disentanglement
                        if "air_image" in out:
                            tb_writer.add_images(config['name'] + "_view_{}/air".format(viewpoint.image_name),
                                                 torch.clamp(out["air_image"], 0.0, 1.0)[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=23, help="Random seed for reproducibility (CPU+CUDA).")
    parser.add_argument('--deterministic', type=int, default=0, help="1 to enable stricter deterministic algorithms (may be slower).")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    # Ensure model_path exists before saving cfg_args
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    os.makedirs(args.model_path, exist_ok = True)
    
    # Extract all parameter groups to ensure complete parameter coverage
    lp_extracted = lp.extract(args)
    op_extracted = op.extract(args)
    pp_extracted = pp.extract(args)
    
    # Merge all parameters into a single namespace for saving
    # Start with the original args (contains all parsed arguments)
    merged_vars = vars(args).copy()
    # Add extracted parameters (ensures all defaults are included even if not explicitly set)
    merged_vars.update(vars(lp_extracted))
    merged_vars.update(vars(op_extracted))
    merged_vars.update(vars(pp_extracted))
    
    # Save complete configuration including PipelineParams and OptimizationParams
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**merged_vars)))

    # Initialize RNG (must happen before any random camera sampling / init).
    setup_seed(int(getattr(args, "seed", 23)))
    _maybe_enable_determinism(bool(int(getattr(args, "deterministic", 0)) == 1))

    # Initialize system state (legacy helper)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp_extracted, op_extracted, pp_extracted, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
