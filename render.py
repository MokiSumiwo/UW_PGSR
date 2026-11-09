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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from scene.medium_model import MediumEstimator
from scene.complete_physics_model import CompleteUnderwaterPhysics
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy

# -------------------- Alpha utility (for bgblack) --------------------
def compute_air_alpha_eff(
    rendered_alpha_1hw: torch.Tensor,
    plane_depth_hw: torch.Tensor,
    *,
    alpha_thr: float = 1e-4,
    use_depth_gate: bool = True,
    z_cut: float = 60.0,
    z_tau: float = 5.0,
    hit_map_hw: torch.Tensor = None,
    scheme4_far_protect: bool = True,
    scheme4_alpha_prot: float = 0.28,
    scheme4_alpha_tau: float = 0.08,
):
    """
    Return alpha_eff (H,W) similar to Scheme4 gating. This is only used as a fallback
    when renderer does not provide alpha_mix.
    """
    a = rendered_alpha_1hw.squeeze(0) if rendered_alpha_1hw.dim() == 3 else rendered_alpha_1hw
    z = plane_depth_hw.squeeze(0) if plane_depth_hw.dim() == 3 else plane_depth_hw
    a = a.clamp(0.0, 1.0)

    hit = (a > float(alpha_thr))
    if hit_map_hw is not None:
        hm = hit_map_hw.squeeze(0) if hit_map_hw.dim() == 3 else hit_map_hw
        if hm.shape == hit.shape:
            hit = hit & (hm > 0)

    a_hit = torch.where(hit, a, torch.zeros_like(a))

    if not use_depth_gate:
        return a_hit

    z_tau = max(float(z_tau), 1e-6)
    far_w = torch.sigmoid((z - float(z_cut)) / z_tau)

    if bool(scheme4_far_protect):
        a_tau = max(float(scheme4_alpha_tau), 1e-6)
        p_alpha = torch.sigmoid((a_hit - float(scheme4_alpha_prot)) / a_tau).clamp(0.0, 1.0)
        gate = (1.0 - far_w * (1.0 - p_alpha)).clamp(0.0, 1.0)
        return (a_hit * gate).clamp(0.0, 1.0)

    return (a_hit * (1.0 - far_w)).clamp(0.0, 1.0)

def apply_bgblack_hard_mask(air_img_3hw: torch.Tensor, bg_mask_hw: torch.Tensor) -> torch.Tensor:
    """
    Hard mask: set background pixels to black, keep foreground untouched.
    This avoids imprinting alpha high-frequency patterns into image brightness.
    """
    out = air_img_3hw.clone()
    out[:, bg_mask_hw] = 0.0
    return out.clamp(0.0, 1.0)

# === OpenEXR 写入工具 ===
import OpenEXR
import Imath

def write_exr_rgb(path, rgb_float32_img):
    img = np.ascontiguousarray(rgb_float32_img, dtype=np.float32)
    assert img.ndim == 3 and img.shape[2] == 3, "EXR expects HxWx3 float32 RGB"
    h, w, _ = img.shape

    header = OpenEXR.Header(w, h)
    try:
        header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    except Exception:
        pass

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    header['channels'] = {
        'R': Imath.Channel(FLOAT),
        'G': Imath.Channel(FLOAT),
        'B': Imath.Channel(FLOAT),
    }

    R = np.ascontiguousarray(img[:, :, 0]).tobytes()
    G = np.ascontiguousarray(img[:, :, 1]).tobytes()
    B = np.ascontiguousarray(img[:, :, 2]).tobytes()

    out = OpenEXR.OutputFile(path, header)
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()

def write_exr_gray(path, gray_float32_img):
    img = np.ascontiguousarray(gray_float32_img, dtype=np.float32)
    assert img.ndim == 2, "EXR gray expects HxW float32 array"
    h, w = img.shape

    header = OpenEXR.Header(w, h)
    try:
        header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    except Exception:
        pass

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    header['channels'] = {'Z': Imath.Channel(FLOAT)}

    Z = np.ascontiguousarray(img).tobytes()
    out = OpenEXR.OutputFile(path, header)
    out.writePixels({'Z': Z})
    out.close()

def linear_to_srgb_np(x: np.ndarray) -> np.ndarray:
    a = 0.055
    t = 0.0031308
    x_clamped = np.clip(x, 0.0, 1.0)
    low = x_clamped * 12.92
    high = (1 + a) * np.power(x_clamped, 1 / 2.4) - a
    return np.where(x_clamped <= t, low, high)

def post_process_mesh(mesh, cluster_to_keep=1):
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background,
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_depth_exr_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth_exr")
    render_air_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_air")
    render_air_raw_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_air_raw")
    render_air_black_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_air_bgblack")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")
    render_normal_exr_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal_exr")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_depth_exr_path, exist_ok=True)
    makedirs(render_air_path, exist_ok=True)
    makedirs(render_air_raw_path, exist_ok=True)
    makedirs(render_air_black_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    makedirs(render_normal_exr_path, exist_ok=True)

    depths_tsdf_fusion = []
    colors_tsdf_rgb = []

    S_colmap_from_ue = torch.tensor([[0, 1, 0],
                                     [0, 0, -1],
                                     [1, 0, 0]], dtype=torch.float32, device="cuda")
    S_ue_from_colmap = S_colmap_from_ue.t()

    def apply_best_signed_permutation(n_xyz: torch.Tensor) -> torch.Tensor:
        x = n_xyz[..., 0]
        y = n_xyz[..., 1]
        z = n_xyz[..., 2]
        out = torch.stack([z, -x, -y], dim=-1)
        return out

    complete_physics_model = None
    medium_model = None

    if pipeline.use_underwater_model:
        try:
            complete_physics_model = CompleteUnderwaterPhysics(appearance_dim=16).cuda()
            complete_physics_model.load_weights(model_path, iteration if iteration != -1 else scene.loaded_iter)
            complete_physics_model.eval()
            print("[Info] Complete physics model loaded successfully")
        except Exception as e:
            print("[Warn] Complete physics model not loaded: {}".format(e))
            complete_physics_model = None
            try:
                medium_model = MediumEstimator(num_basis=pipeline.medium_num_basis, use_spatial_smooth=True).cuda()
                medium_model.load_weights(model_path, iteration if iteration != -1 else scene.loaded_iter)
                medium_model.eval()
                print("[Info] Fallback to original medium model")
            except Exception as e:
                print("[Warn] medium_model not loaded: {}".format(e))
                medium_model = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()

        out = render(
            view, gaussians, pipeline, background,
            app_model=app_model,
            medium_model=medium_model,
            complete_physics_model=complete_physics_model
        )

        rendering = out["render"].clamp(0.0, 1.0)

        air_img_raw = out.get("air_image", rendering)
        air_img = out.get("air_image_closed", air_img_raw).clamp(0.0, 1.0)

        _, H, W = rendering.shape

        render_rgb_np = (rendering.permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
        colors_tsdf_rgb.append(render_rgb_np)

        depth = out["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth_np = depth.detach().cpu().numpy()
        depth01 = np.clip(depth_np / 1000.0, 0.0, 1.0)

        # -------------------- bgblack mask: use alpha_mix first --------------------
        bg_mask_t = None
        bg_mask_np = None
        a_eff_t = None
        if bool(getattr(pipeline, "air_bg_black", 0)):
            thr = float(getattr(pipeline, "air_bg_depth_one_thr", 1e-3))

            # Prefer renderer-provided alpha_mix (already hit-masked and Scheme4 gated)
            alpha_mix = out.get("alpha_mix", None)
            if alpha_mix is not None:
                a_eff_t = alpha_mix.squeeze(0) if alpha_mix.dim() == 3 else alpha_mix
                a_eff_t = a_eff_t.clamp(0.0, 1.0)
            else:
                # Fallback: reproduce alpha_eff from rendered_alpha + depth
                a = out.get("rendered_alpha", None)
                z = out.get("plane_depth", None)
                hm = out.get("hit_map", None)
                if a is not None and z is not None:
                    alpha_thr = float(getattr(pipeline, "air_bg_alpha_thr", -1.0))
                    if alpha_thr <= 0:
                        alpha_thr = float(getattr(pipeline, "physics_alpha_thr", 1e-4))

                    z_cut = float(getattr(pipeline, "air_bg_z_cut", -1.0))
                    if z_cut <= 0:
                        z_cut = float(getattr(pipeline, "bg_z_cut", 60.0))

                    z_tau = float(getattr(pipeline, "air_bg_z_tau", -1.0))
                    if z_tau <= 0:
                        z_tau = float(getattr(pipeline, "bg_z_tau", 5.0))

                    use_gate = bool(getattr(pipeline, "air_bg_use_depth_gate", 1)) and bool(getattr(pipeline, "physics_depth_gated_background", 0))
                    far_protect = bool(getattr(pipeline, "scheme4_far_protect", 1))
                    a_prot = float(getattr(pipeline, "scheme4_alpha_prot", 0.28))
                    a_tau = float(getattr(pipeline, "scheme4_alpha_tau", 0.08))

                    a_eff_t = compute_air_alpha_eff(
                        a, z,
                        alpha_thr=alpha_thr,
                        use_depth_gate=use_gate,
                        z_cut=z_cut,
                        z_tau=z_tau,
                        hit_map_hw=hm if bool(getattr(pipeline, "air_bg_use_hit_map", 0)) else None,
                        scheme4_far_protect=far_protect,
                        scheme4_alpha_prot=a_prot,
                        scheme4_alpha_tau=a_tau,
                    )

            if a_eff_t is not None:
                bg_mask_t = (a_eff_t < thr)
                bg_mask_np = bg_mask_t.detach().cpu().numpy()

        # 若开启 air_bg_black 且要求 depth 填充为 1.0
        if bool(getattr(pipeline, "air_bg_black", 0)) and bool(getattr(pipeline, "air_bg_depth_fill_one", 1)):
            try:
                if bg_mask_np is not None:
                    depth01[bg_mask_np] = 1.0
            except Exception:
                pass

        depth_exr_path = os.path.join(render_depth_exr_path, view.image_name + ".exr")
        write_exr_gray(depth_exr_path, depth01)

        depth_i = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        # ========= 法线从 COLMAP 相机系 -> UE 相机系 =========
        normal_cam_colmap = out["rendered_normal"].permute(1, 2, 0)

        if normal_cam_colmap.dtype != S_ue_from_colmap.dtype:
            normal_cam_colmap = normal_cam_colmap.to(dtype=S_ue_from_colmap.dtype)
        if normal_cam_colmap.device != S_ue_from_colmap.device:
            normal_cam_colmap = normal_cam_colmap.to(device=S_ue_from_colmap.device)

        normal_cam_ue = torch.matmul(normal_cam_colmap, S_ue_from_colmap)
        normal_cam_ue = normal_cam_ue / (normal_cam_ue.norm(dim=-1, keepdim=True) + 1.0e-8)

        normal_aligned = apply_best_signed_permutation(normal_cam_ue)
        normal_aligned = normal_aligned / (normal_aligned.norm(dim=-1, keepdim=True) + 1.0e-8)

        # nohit 或 bgblack 区域法线填充
        try:
            hm = out.get("hit_map", None)
            if hm is not None:
                nohit_mask = (hm.squeeze(0) if hm.dim() == 3 else hm) <= 0
            else:
                nohit_mask = depth <= 0

            fill_mask = nohit_mask
            if bg_mask_t is not None:
                fill_mask = fill_mask | bg_mask_t.to(device=normal_aligned.device)

            if fill_mask is not None and torch.is_tensor(fill_mask) and fill_mask.shape == (H, W):
                fill_dir = torch.tensor([0.0, 0.0, -1.0], device=normal_aligned.device, dtype=normal_aligned.dtype).view(1, 1, 3)
                fill_dir = fill_dir.expand_as(normal_aligned)
                normal_aligned = torch.where(fill_mask.unsqueeze(-1), fill_dir, normal_aligned)
        except Exception:
            pass

        normal_01 = torch.clamp((normal_aligned + 1.0) * 0.5, 0.0, 1.0)
        normal_exr_rgb = normal_01.detach().cpu().numpy().astype(np.float32)
        normal_vis_rgb = (linear_to_srgb_np(normal_exr_rgb) * 255.0 + 0.5).astype(np.uint8)

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1, 2, 0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0

        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())

        render_bgr_up = render_rgb_np[:, :, ::-1]
        gt_np = (gt.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        gt_bgr_up = gt_np[:, :, ::-1]
        normal_vis_bgr_up = normal_vis_rgb[:, :, ::-1]

        # -------------------- write images --------------------
        is_test = (name == "test")
        ext = ".png" if is_test else ".jpg"

        cv2.imwrite(os.path.join(render_path, view.image_name + ext), render_bgr_up)
        if is_test:
            cv2.imwrite(os.path.join(gts_path, view.image_name + ext), gt_bgr_up)

        air_raw_bgr = (air_img_raw.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(render_air_raw_path, view.image_name + ext), air_raw_bgr)

        air_bgr = (air_img.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(render_air_path, view.image_name + ext), air_bgr)

        # bgblack: hard-mask to black using the SAME alpha_mix logic, to avoid stripe imprinting
        if bool(getattr(pipeline, "air_bg_black", 0)) and bg_mask_t is not None:
            air_black = apply_bgblack_hard_mask(air_img, bg_mask_t)
            air_black_bgr = (air_black.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(render_air_black_path, view.image_name + ext), air_black_bgr)

        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".png"), depth_color)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".png"), normal_vis_bgr_up)

        exr_path = os.path.join(render_normal_exr_path, view.image_name + ".exr")
        write_exr_rgb(exr_path, normal_exr_rgb)

    # TSDF fusion
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()
            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth > max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()

            pose = np.identity(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T

            color_rgb = np.ascontiguousarray(colors_tsdf_rgb[idx])
            color_o3d = o3d.geometry.Image(color_rgb)

            depth_u16 = np.ascontiguousarray((ref_depth * 1000).astype(np.uint16))
            depth_o3d = o3d.geometry.Image(depth_u16)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False
            )

            H_render, W_render = color_rgb.shape[0], color_rgb.shape[1]
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W_render, H_render, view.Fx, view.Fy, view.Cx, view.Cy),
                pose
            )

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                max_depth: float, voxel_size: float, num_cluster: int, use_depth_filter: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        app_model = AppModel()
        app_model.load_weights(scene.model_path)
        app_model.eval()
        app_model.cuda()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        if not skip_train:
            render_set(
                dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                scene, gaussians, pipeline, background,
                app_model=app_model, max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter
            )
            print("extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "mesh")
            os.makedirs(path, exist_ok=True)

            o3d.io.write_triangle_mesh(
                os.path.join(path, "tsdf_fusion.ply"),
                mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True
            )

            mesh = post_process_mesh(mesh, num_cluster)
            o3d.io.write_triangle_mesh(
                os.path.join(path, "tsdf_fusion_post.ply"),
                mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True
            )

        if not skip_test:
            render_set(
                dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                scene, gaussians, pipeline, background,
                app_model=app_model
            )

if __name__ == "__main__":
    torch.set_num_threads(8)
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    print("multi_view_num {}".format(model.multi_view_num))
    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args),
        args.skip_train, args.skip_test, args.max_depth,
        args.voxel_size, args.num_cluster, args.use_depth_filter
    )