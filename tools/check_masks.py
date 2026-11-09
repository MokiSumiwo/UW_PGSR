import os
import sys
import json
from argparse import ArgumentParser
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make script runnable from any working directory:
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.app_model import AppModel
from scene.complete_physics_model import CompleteUnderwaterPhysics
from utils.general_utils import safe_state


def _to_uint8(rgb_3hw: torch.Tensor) -> np.ndarray:
    x = rgb_3hw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def _save_gray01(path: str, x_hw: torch.Tensor):
    x = x_hw.detach()
    if x.dim() == 3:
        x = x.squeeze(0)
    x = x.float()
    flat = x.reshape(-1)
    if flat.numel() > 0:
        try:
            lo = torch.quantile(flat, 0.01).item()
            hi = torch.quantile(flat, 0.99).item()
        except Exception:
            lo, hi = float(flat.min().item()), float(flat.max().item())
    else:
        lo, hi = 0.0, 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-8:
        lo, hi = float(x.min().item()), float(x.max().item())
        if (hi - lo) < 1e-8:
            lo, hi = 0.0, 1.0
    y = ((x - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    img = (y.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(path, img)


def _save_mask(path: str, m_hw: torch.Tensor):
    m = m_hw.detach()
    if m.dim() == 3:
        m = m.squeeze(0)
    if m.dtype != torch.bool:
        m = (m > 0.5)
    img = (m.cpu().numpy().astype(np.uint8) * 255)
    cv2.imwrite(path, img)


def _overlay(base_rgb_3hw: torch.Tensor, blue_hw: Optional[torch.Tensor], red_hw: Optional[torch.Tensor]) -> np.ndarray:
    base = base_rgb_3hw.detach().clamp(0.0, 1.0)
    out = base.clone()
    H, W = int(out.shape[1]), int(out.shape[2])

    if blue_hw is not None:
        mb = blue_hw.detach()
        if mb.dim() == 3:
            mb = mb.squeeze(0)
        mb = (mb > 0.5)
        if mb.shape == (H, W):
            blue = torch.tensor([0.0, 0.0, 1.0], device=out.device, dtype=out.dtype).view(3, 1)
            out[:, mb] = 0.45 * out[:, mb] + 0.55 * blue

    if red_hw is not None:
        mr = red_hw.detach()
        if mr.dim() == 3:
            mr = mr.squeeze(0)
        mr = (mr > 0.5)
        if mr.shape == (H, W):
            red = torch.tensor([1.0, 0.0, 0.0], device=out.device, dtype=out.dtype).view(3, 1)
            out[:, mr] = 0.25 * out[:, mr] + 0.75 * red

    return _to_uint8(out)


@torch.no_grad()
def build_bg_mask_gt_alpha_no_conn(
    gt_3hw: torch.Tensor,
    gt_gray_1hw: torch.Tensor,
    alpha_hw: torch.Tensor,
    binf_3: Optional[torch.Tensor],
    *,
    alpha_bg_thr: float = 0.10,
    alpha_gamma: float = 1.5,
    bg_gt_w: float = 0.58,
    seed_gray_min: float = 0.03,
    seed_alpha_thr: float = 0.03,
    seed_binf_eps: float = 0.35,
    min_seed_px: int = 512,
    binf_calib_w: float = 0.5,
    binf_eps: float = 0.25,
    use_binf_gate: bool = True,
    sigma_col_min: float = 0.02,
    sigma_col_max: float = 0.15,
    col_thr: float = 0.35,
    morph_k: int = 5,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict]:
    device = gt_3hw.device
    _, H, W = gt_3hw.shape

    a = alpha_hw.squeeze(0) if alpha_hw.dim() == 3 else alpha_hw
    a = a.clamp(0.0, 1.0)

    gray = gt_gray_1hw.squeeze(0) if gt_gray_1hw.dim() == 3 else gt_gray_1hw
    gray = gray.clamp(0.0, 1.0)
    valid = (gray > float(seed_gray_min))

    eps = 1e-6
    bg_gt_w = float(max(0.0, min(1.0, bg_gt_w)))

    thr = float(max(alpha_bg_thr, 1e-6))
    w_alpha = ((thr - a) / thr).clamp(0.0, 1.0)
    w_alpha = (w_alpha ** float(max(alpha_gamma, 1e-6))) * valid.float()

    # Fallback if no B_inf
    if (binf_3 is None) or (binf_3.numel() != 3):
        seed = valid & (a < min(float(seed_alpha_thr), float(alpha_bg_thr)))
        if seed.sum().item() < 64 and valid.any():
            seed = valid & (a < float(alpha_bg_thr))
        if seed.sum().item() < 64:
            seed = valid

        vals = gt_3hw[:, seed]
        B_ref = torch.quantile(vals, 0.5, dim=1).clamp(0.0, 1.0).view(3, 1, 1)

        dist = (gt_3hw - B_ref).abs().mean(0)
        sigma = float(torch.std(vals, dim=1).mean().item())
        sigma = max(float(sigma_col_min), min(float(sigma_col_max), sigma))
        w_col = torch.exp(-(dist * dist) / (2.0 * sigma * sigma)).clamp(0.0, 1.0) * valid.float()

        bg_soft = torch.exp(bg_gt_w * torch.log(w_col + eps) + (1.0 - bg_gt_w) * torch.log(w_alpha + eps))
        bg_soft = (bg_soft > float(col_thr)).float().unsqueeze(0)

        k = int(morph_k)
        if k > 1:
            if k % 2 == 0:
                k += 1
            bg_soft = F.avg_pool2d(bg_soft.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0).clamp(0.0, 1.0)

        st = {"skip": 0, "use_binf": 0, "mask_mean": float(bg_soft.mean().item()), "seed_pixels": int(seed.sum().item())}
        return bg_soft, B_ref.view(3).detach(), st

    # With B_inf
    binf = binf_3.view(3, 1, 1).to(gt_3hw.device, gt_3hw.dtype)
    dist_binf = (gt_3hw - binf).abs().mean(0)

    seed = valid & (dist_binf < float(seed_binf_eps)) & (a < float(seed_alpha_thr))
    if seed.sum().item() < 64:
        seed = valid & (dist_binf < float(seed_binf_eps)) & (a < float(alpha_bg_thr))

    if seed.sum().item() < int(min_seed_px):
        cand = valid
        score = (1.0 - torch.clamp(dist_binf / max(float(seed_binf_eps), 1e-6), 0.0, 1.0)) * (1.0 - a)
        score = score * cand.float()
        flat_score = score.reshape(-1)
        flat_cand = cand.reshape(-1)
        idx = torch.nonzero(flat_cand, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            k = min(int(min_seed_px), int(idx.numel()))
            top = torch.topk(flat_score[idx], k=k, largest=True).indices
            pick = idx[top]
            seed_flat = torch.zeros((H * W,), device=device, dtype=torch.bool)
            seed_flat[pick] = True
            seed = seed_flat.view(H, W)

    n_seed = int(seed.sum().item())
    if n_seed < 32:
        return None, None, {"skip": 1, "reason": "few_seed", "seed_pixels": n_seed}

    vals = gt_3hw[:, seed]
    B_gt = torch.quantile(vals, 0.5, dim=1).clamp(0.0, 1.0)

    w = max(0.0, min(1.0, float(binf_calib_w)))
    B_ref = ((1.0 - w) * binf_3.detach().clamp(0.0, 1.0) + w * B_gt.detach().clamp(0.0, 1.0)).view(3, 1, 1)

    dist = (gt_3hw - B_ref).abs().mean(0)
    sigma = float(torch.std(vals, dim=1).mean().item())
    sigma = max(float(sigma_col_min), min(float(sigma_col_max), sigma))
    w_col = torch.exp(-(dist * dist) / (2.0 * sigma * sigma)).clamp(0.0, 1.0) * valid.float()

    if bool(use_binf_gate):
        w_col = w_col * (dist < float(binf_eps)).float()

    bg = torch.exp(bg_gt_w * torch.log(w_col + eps) + (1.0 - bg_gt_w) * torch.log(w_alpha + eps))
    bg_soft = bg.unsqueeze(0).clamp(0.0, 1.0)

    k = int(morph_k)
    if k > 1:
        if k % 2 == 0:
            k += 1
        bg_soft = F.avg_pool2d(bg_soft.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0).clamp(0.0, 1.0)

    st = {
        "skip": 0,
        "use_binf": 1,
        "seed_pixels": int(n_seed),
        "sigma_col": float(sigma),
        "mask_mean": float(bg_soft.mean().item()),
        "alpha_bg_thr": float(alpha_bg_thr),
        "alpha_gamma": float(alpha_gamma),
        "bg_gt_w": float(bg_gt_w),
        "seed_binf_eps": float(seed_binf_eps),
        "binf_calib_w": float(binf_calib_w),
        "binf_eps": float(binf_eps),
        "use_binf_gate": bool(use_binf_gate),
    }
    return bg_soft, B_ref.view(3).detach(), st


@torch.no_grad()
def main():
    parser = ArgumentParser("Mask checker (GT + alpha-black + B_inf; no connectivity)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_views", type=int, default=-1)
    parser.add_argument("--dump_debug", type=int, default=1)

    parser.add_argument("--mask_thr", type=float, default=0.35, help="bg_soft>thr => bg_bin for overlay.")

    # 关键改动：这里不再叫“cover”，而是“高斯存在阈值”
    parser.add_argument(
        "--cover_alpha_thr",
        type=float,
        default=1e-4,
        help="Gaussian presence threshold: if rendered_alpha > thr, we consider a Gaussian contributes on that pixel."
    )

    parser.add_argument("--alpha_bg_thr", type=float, default=0.10)
    parser.add_argument("--alpha_gamma", type=float, default=1.5)
    parser.add_argument("--bg_gt_w", type=float, default=0.58)

    parser.add_argument("--seed_gray_min", type=float, default=0.03)
    parser.add_argument("--seed_alpha_thr", type=float, default=0.03)
    parser.add_argument("--seed_binf_eps", type=float, default=0.35)
    parser.add_argument("--min_seed_px", type=int, default=512)
    parser.add_argument("--binf_calib_w", type=float, default=0.5)

    parser.add_argument("--use_binf_ref", type=int, default=1)
    parser.add_argument("--use_binf_gate", type=int, default=1)
    parser.add_argument("--binf_eps", type=float, default=0.25)

    parser.add_argument("--sigma_col_min", type=float, default=0.02)
    parser.add_argument("--sigma_col_max", type=float, default=0.15)
    parser.add_argument("--col_thr", type=float, default=0.35)
    parser.add_argument("--morph_k", type=int, default=5)

    args = get_combined_args(parser)
    safe_state(getattr(args, "quiet", False))

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    out_root = os.path.abspath(args.out_dir)
    out_overlay = os.path.join(out_root, "gt_alpha_bgmask")
    out_dbg = os.path.join(out_root, "debug_maps")
    os.makedirs(out_overlay, exist_ok=True)
    if int(getattr(args, "dump_debug", 1)) == 1:
        os.makedirs(out_dbg, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=int(args.iteration), shuffle=False)

    app_model = AppModel().cuda().eval()
    try:
        app_model.load_weights(dataset.model_path, scene.loaded_iter)
    except TypeError:
        app_model.load_weights(dataset.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    binf_3 = None
    if int(getattr(args, "use_binf_ref", 1)) == 1:
        try:
            physics = CompleteUnderwaterPhysics(appearance_dim=16).cuda().eval()
            physics.load_weights(dataset.model_path, scene.loaded_iter)
            if hasattr(physics, "backscatter_net") and hasattr(physics.backscatter_net, "B_inf"):
                b = physics.backscatter_net.B_inf.detach()
                b = b.view(3) if b.numel() == 3 else b.view(-1)[:3]
                binf_3 = b.clamp(0.0, 1.0).to(device="cuda", dtype=torch.float32)
        except Exception:
            binf_3 = None

    try:
        print("B_inf loaded:", binf_3 is not None, "value:", None if binf_3 is None else [float(x) for x in binf_3.tolist()])
    except Exception:
        pass

    views = scene.getTrainCameras()
    if int(args.max_views) > 0:
        views = views[: int(args.max_views)]

    mask_thr = float(args.mask_thr)
    presence_thr = float(args.cover_alpha_thr)

    stats_path = os.path.join(out_root, "per_view_stats.jsonl")
    if int(getattr(args, "dump_debug", 1)) == 1:
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass

    for cam in tqdm(views, desc="Rendering train views"):
        gt_3hw, gt_gray_1hw = cam.get_image()
        if gt_gray_1hw.dim() == 2:
            gt_gray_1hw = gt_gray_1hw.unsqueeze(0)

        pkg = render(
            cam,
            gaussians,
            pipe,
            background,
            app_model=app_model,
            medium_model=None,
            complete_physics_model=None,
            # Need rendered_alpha -> only available in geometry/plane branch.
            # We still do NOT use depth/hit_map here; this is only to get alpha.
            return_plane=True,
            return_depth_normal=False,
        )

        air = pkg.get("air_image", pkg["render"]).clamp(0.0, 1.0)
        if "rendered_alpha" not in pkg:
            # Robustness: skip this view if renderer didn't return alpha for any reason.
            # (Should not happen when return_plane=True.)
            continue
        alpha = pkg["rendered_alpha"].squeeze(0).clamp(0.0, 1.0)

        bg_soft, B_ref, st = build_bg_mask_gt_alpha_no_conn(
            gt_3hw, gt_gray_1hw, alpha, binf_3,
            alpha_bg_thr=float(args.alpha_bg_thr),
            alpha_gamma=float(args.alpha_gamma),
            bg_gt_w=float(args.bg_gt_w),
            seed_gray_min=float(args.seed_gray_min),
            seed_alpha_thr=float(args.seed_alpha_thr),
            seed_binf_eps=float(args.seed_binf_eps),
            min_seed_px=int(args.min_seed_px),
            binf_calib_w=float(args.binf_calib_w),
            binf_eps=float(args.binf_eps),
            use_binf_gate=bool(int(args.use_binf_gate) == 1),
            sigma_col_min=float(args.sigma_col_min),
            sigma_col_max=float(args.sigma_col_max),
            col_thr=float(args.col_thr),
            morph_k=int(args.morph_k),
        )

        if bg_soft is None:
            img = _to_uint8(air)
        else:
            bg_bin = (bg_soft.squeeze(0) > mask_thr)

            # 关键改动：只要存在高斯贡献（alpha > presence_thr），背景像素就标红
            gauss_present = (alpha > presence_thr)
            red = bg_bin & gauss_present
            blue = bg_bin & (~red)

            img = _overlay(air, blue, red)

        cv2.imwrite(os.path.join(out_overlay, f"{cam.image_name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if int(getattr(args, "dump_debug", 1)) == 1:
            stem = cam.image_name
            _save_gray01(os.path.join(out_dbg, f"{stem}_alpha.png"), alpha)

            if bg_soft is not None:
                _save_gray01(os.path.join(out_dbg, f"{stem}_bg_soft.png"), bg_soft.squeeze(0))
                _save_mask(os.path.join(out_dbg, f"{stem}_bg_bin.png"), bg_bin)
                _save_mask(os.path.join(out_dbg, f"{stem}_gauss_present.png"), gauss_present)
                _save_mask(os.path.join(out_dbg, f"{stem}_bg_red.png"), (bg_bin & gauss_present))
                _save_mask(os.path.join(out_dbg, f"{stem}_bg_blue.png"), (bg_bin & (~gauss_present)))

            if binf_3 is not None:
                d = (gt_3hw - binf_3.view(3, 1, 1).to(gt_3hw.device, gt_3hw.dtype)).abs().mean(0)
                _save_gray01(os.path.join(out_dbg, f"{stem}_dist_binf.png"), d)

            try:
                rec = {"name": stem, **st, "presence_thr": float(presence_thr)}
                if B_ref is not None and isinstance(B_ref, torch.Tensor):
                    rec["B_ref"] = [float(x) for x in B_ref.detach().flatten().tolist()]
                with open(stats_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

    try:
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_path": dataset.model_path,
                    "loaded_iter": int(scene.loaded_iter),
                    "num_views": int(len(views)),
                    "mask_thr": float(mask_thr),
                    "gauss_presence_thr": float(presence_thr),
                    "alpha_bg_thr": float(args.alpha_bg_thr),
                    "alpha_gamma": float(args.alpha_gamma),
                    "bg_gt_w": float(args.bg_gt_w),
                    "use_binf_ref": int(getattr(args, "use_binf_ref", 1)),
                    "use_binf_gate": int(getattr(args, "use_binf_gate", 1)),
                    "binf_eps": float(args.binf_eps),
                    "seed_binf_eps": float(args.seed_binf_eps),
                    "binf_calib_w": float(args.binf_calib_w),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
