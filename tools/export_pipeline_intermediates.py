import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.app_model import AppModel
from scene.complete_physics_model import CompleteUnderwaterPhysics
from utils.general_utils import safe_state


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _chw_to_hwc(x):
    if x.ndim == 3 and x.shape[0] in (1, 3):
        return np.transpose(x, (1, 2, 0))
    return x


def _save_rgb(path, tensor_or_array, srgb=False):
    arr = _to_numpy(tensor_or_array)
    arr = _chw_to_hwc(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    if srgb:
        a = 0.055
        t = 0.0031308
        arr = np.where(arr <= t, arr * 12.92, (1 + a) * np.power(arr, 1 / 2.4) - a)
    bgr = (arr[..., :3] * 255.0 + 0.5).astype(np.uint8)[..., ::-1]
    cv2.imwrite(path, bgr)


def _normalize_valid(arr, valid=None, lo_q=1.0, hi_q=99.0):
    arr = np.asarray(arr, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(arr)
    else:
        valid = valid & np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[valid]
    lo = np.percentile(vals, lo_q)
    hi = np.percentile(vals, hi_q)
    if hi <= lo:
        hi = lo + 1e-6
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = 0.0
    return out.astype(np.float32)


def _save_gray(path, tensor_or_array, valid=None, invert=False):
    arr = _to_numpy(tensor_or_array).squeeze()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = _normalize_valid(arr, valid=valid)
    else:
        arr = np.clip(arr, 0.0, 1.0)
        if valid is not None:
            arr = arr.copy()
            arr[~valid] = 0.0
    if invert:
        arr = 1.0 - arr
        if valid is not None:
            arr[~valid] = 0.0
    cv2.imwrite(path, (arr * 255.0 + 0.5).astype(np.uint8))


def _save_heat(path, tensor_or_array, valid=None, invert=False):
    arr = _to_numpy(tensor_or_array).squeeze()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = _normalize_valid(arr, valid=valid)
    else:
        arr = np.clip(arr, 0.0, 1.0)
        if valid is not None:
            arr = arr.copy()
            arr[~valid] = 0.0
    if invert:
        arr = 1.0 - arr
        if valid is not None:
            arr[~valid] = 0.0
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    if valid is not None:
        color[~valid] = 0
    cv2.imwrite(path, color)


def _save_mask(path, tensor_or_array):
    arr = _to_numpy(tensor_or_array).squeeze()
    arr = (arr > 0.5).astype(np.uint8) * 255
    cv2.imwrite(path, arr)


def _save_npy(path, tensor_or_array):
    np.save(path, _to_numpy(tensor_or_array))


def _find_view(scene, target):
    target = os.path.splitext(os.path.basename(str(target)))[0]
    candidates = [target]
    try:
        candidates.append(f"{int(target):04d}")
    except Exception:
        pass
    cameras = list(scene.getTestCameras()) + list(scene.getTrainCameras())
    for name in candidates:
        for view in cameras:
            if view.image_name == name:
                return view
    names = [v.image_name for v in cameras]
    raise RuntimeError(f"View {target!r} not found. Available examples: {names[:20]}")


def main():
    parser = ArgumentParser(description="Export GeoMed-GS pipeline intermediate visualizations.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--view", default="0104", type=str)
    parser.add_argument("--out_dir", default="", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        view = _find_view(scene, args.view)

        app_model = AppModel()
        try:
            app_model.load_weights(scene.model_path, scene.loaded_iter)
            app_model.eval().cuda()
        except Exception as exc:
            print(f"[warn] app_model not loaded: {exc}")
            app_model = None

        complete_physics_model = None
        if bool(getattr(pipe, "use_underwater_model", False)):
            complete_physics_model = CompleteUnderwaterPhysics(
                appearance_dim=16,
                max_depth=float(getattr(pipe, "physics_net_max_depth", 100.0)),
            ).cuda()
            complete_physics_model.load_weights(scene.model_path, scene.loaded_iter)
            complete_physics_model.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = render(
            view,
            gaussians,
            pipe,
            background,
            app_model=app_model,
            complete_physics_model=complete_physics_model,
        )

        out_dir = args.out_dir or os.path.join(
            dataset.model_path,
            "pipeline_intermediates",
            f"iter_{scene.loaded_iter}",
            view.image_name,
        )
        os.makedirs(out_dir, exist_ok=True)
        raw_dir = os.path.join(out_dir, "raw_npy")
        os.makedirs(raw_dir, exist_ok=True)

        render_img = out["render"].clamp(0.0, 1.0)
        clean_img = out["air_image"].clamp(0.0, 1.0)
        alpha_raw = out["rendered_alpha"].squeeze(0)
        alpha = alpha_raw.clamp(0.0, 1.0)
        depth = out["plane_depth"].squeeze()
        hit = (alpha_raw > float(getattr(pipe, "physics_alpha_thr", 1e-4)))
        hit_map = out.get("hit_map", hit.float()).squeeze()
        hit_combined = hit & (hit_map > 0)

        z_cut = float(out.get("scheme4_z_cut", torch.tensor(float(getattr(pipe, "bg_z_cut", 60.0)))).detach().cpu().item())
        z_tau = max(float(getattr(pipe, "bg_z_tau", 5.0)), 1e-6)
        far_gate = out.get("far_gate", torch.sigmoid((depth - z_cut) / z_tau)).squeeze().clamp(0.0, 1.0)
        a_tau = max(float(getattr(pipe, "scheme4_alpha_tau", 0.08)), 1e-6)
        a_prot = float(getattr(pipe, "scheme4_alpha_prot", 0.28))
        alpha_hit = torch.where(hit, alpha_raw, torch.zeros_like(alpha_raw))
        support_confidence = out.get(
            "support_confidence",
            torch.sigmoid((alpha_hit - a_prot) / a_tau),
        ).squeeze().clamp(0.0, 1.0)
        alpha_mix_gate = out.get(
            "alpha_mix_gate",
            1.0 - far_gate * (1.0 - support_confidence),
        ).squeeze().clamp(0.0, 1.0)
        alpha_mix = out.get(
            "alpha_mix",
            alpha_hit * alpha_mix_gate,
        ).squeeze().clamp(0.0, 1.0)

        physics_depth = out.get("physics_depth", torch.where(hit, depth.clamp_min(1e-3), torch.full_like(depth, float(getattr(pipe, "physics_bg_depth", 1e6)))))
        physics_inv_depth = out.get("physics_inv_depth", (1.0 / (physics_depth + 1e-6)) * hit.float()).clamp(0.0, 2.0)
        attenuation = out.get("attenuation", None)
        backscatter = out.get("backscatter", None)
        if attenuation is None or backscatter is None:
            raise RuntimeError("Renderer did not return attenuation/backscatter. Check complete physics weights.")
        B_inf_param = out.get("B_inf", complete_physics_model.backscatter_net.B_inf).detach()
        B_inf_raw = B_inf_param.expand_as(render_img)
        B_inf_vis = B_inf_raw.clamp(0.0, 1.0)
        I_fg = out.get("foreground_image", clean_img * attenuation + backscatter).clamp(0.0, 1.0)
        alpha_mix_3 = alpha_mix.unsqueeze(0)
        backscatter_foreground = alpha_mix_3 * backscatter
        foreground_component = alpha_mix_3 * I_fg
        background_component = (1.0 - alpha_mix_3) * B_inf_raw
        closed_sum = (foreground_component + background_component).clamp(0.0, 1.0)
        closed_sum_abs_error = torch.mean(torch.abs(closed_sum - render_img), dim=0)

        valid_depth = _to_numpy(hit_combined).astype(bool)
        _save_rgb(os.path.join(out_dir, "0104_underwater_render.png"), render_img)
        _save_rgb(os.path.join(out_dir, "0104_clean_render_J.png"), clean_img)
        _save_rgb(os.path.join(out_dir, "0104_foreground_I_fg.png"), I_fg)
        _save_rgb(os.path.join(out_dir, "0104_alpha_mix_times_I_fg.png"), foreground_component)
        _save_rgb(os.path.join(out_dir, "0104_one_minus_alpha_mix_times_B_inf.png"), background_component)
        _save_rgb(os.path.join(out_dir, "0104_closed_composition_sum.png"), closed_sum)
        _save_heat(os.path.join(out_dir, "0104_closed_composition_abs_error_heat.png"), closed_sum_abs_error)
        _save_rgb(os.path.join(out_dir, "0104_attenuation_A_rgb.png"), attenuation)
        _save_rgb(os.path.join(out_dir, "0104_backscatter_B_rgb.png"), backscatter_foreground)
        _save_rgb(os.path.join(out_dir, "0104_B_inf_tile.png"), B_inf_vis)

        _save_gray(os.path.join(out_dir, "0104_alpha_opacity_gray.png"), alpha)
        _save_heat(os.path.join(out_dir, "0104_alpha_opacity_heat.png"), alpha)
        _save_mask(os.path.join(out_dir, "0104_hit_mask.png"), hit_combined)
        _save_heat(os.path.join(out_dir, "0104_depth_z_heat.png"), depth, valid=valid_depth)
        _save_heat(os.path.join(out_dir, "0104_hit_aware_depth_heat.png"), physics_depth, valid=valid_depth)
        _save_heat(os.path.join(out_dir, "0104_hit_aware_inv_depth_heat.png"), physics_inv_depth, valid=valid_depth)
        _save_gray(os.path.join(out_dir, "0104_far_gate_gray.png"), far_gate)
        _save_heat(os.path.join(out_dir, "0104_far_gate_heat.png"), far_gate)
        _save_gray(os.path.join(out_dir, "0104_support_confidence_gray.png"), support_confidence)
        _save_heat(os.path.join(out_dir, "0104_support_confidence_heat.png"), support_confidence)
        _save_gray(os.path.join(out_dir, "0104_alpha_mix_gray.png"), alpha_mix)
        _save_heat(os.path.join(out_dir, "0104_alpha_mix_heat.png"), alpha_mix)
        _save_gray(os.path.join(out_dir, "0104_alpha_mix_gate_gray.png"), alpha_mix_gate)

        _save_npy(os.path.join(raw_dir, "alpha_opacity.npy"), alpha)
        _save_npy(os.path.join(raw_dir, "rendered_alpha_raw.npy"), alpha_raw)
        _save_npy(os.path.join(raw_dir, "hit_mask.npy"), hit_combined.float())
        _save_npy(os.path.join(raw_dir, "depth_z.npy"), depth)
        _save_npy(os.path.join(raw_dir, "physics_depth.npy"), physics_depth)
        _save_npy(os.path.join(raw_dir, "physics_inv_depth.npy"), physics_inv_depth)
        _save_npy(os.path.join(raw_dir, "attenuation.npy"), attenuation)
        _save_npy(os.path.join(raw_dir, "backscatter.npy"), backscatter)
        _save_npy(os.path.join(raw_dir, "backscatter_alpha_mix_masked.npy"), backscatter_foreground)
        _save_npy(os.path.join(raw_dir, "B_inf.npy"), complete_physics_model.backscatter_net.B_inf.detach())
        _save_npy(os.path.join(raw_dir, "far_gate.npy"), far_gate)
        _save_npy(os.path.join(raw_dir, "support_confidence.npy"), support_confidence)
        _save_npy(os.path.join(raw_dir, "alpha_mix.npy"), alpha_mix)
        _save_npy(os.path.join(raw_dir, "alpha_mix_gate.npy"), alpha_mix_gate)
        _save_npy(os.path.join(raw_dir, "foreground_component_alpha_mix_I_fg.npy"), foreground_component)
        _save_npy(os.path.join(raw_dir, "background_component_one_minus_alpha_mix_B_inf.npy"), background_component)
        _save_npy(os.path.join(raw_dir, "closed_composition_sum.npy"), closed_sum)
        _save_npy(os.path.join(raw_dir, "closed_composition_abs_error.npy"), closed_sum_abs_error)

        with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
            f.write(f"model_path: {dataset.model_path}\n")
            f.write(f"source_path: {dataset.source_path}\n")
            f.write(f"iteration: {scene.loaded_iter}\n")
            f.write(f"view: {view.image_name}\n")
            f.write(f"z_cut: {z_cut}\n")
            f.write(f"z_tau: {z_tau}\n")
            f.write(f"physics_alpha_thr: {float(getattr(pipe, 'physics_alpha_thr', 1e-4))}\n")
            f.write(f"scheme4_alpha_prot: {a_prot}\n")
            f.write(f"scheme4_alpha_tau: {a_tau}\n")
            f.write(f"B_inf_rgb: {_to_numpy(complete_physics_model.backscatter_net.B_inf.detach()).reshape(3).tolist()}\n")
            f.write(f"closed_sum_mean_abs_error: {float(closed_sum_abs_error.mean().item())}\n")
            f.write(f"closed_sum_max_abs_error: {float(closed_sum_abs_error.max().item())}\n")

        print(f"[done] exported pipeline intermediates to: {out_dir}")


if __name__ == "__main__":
    main()
