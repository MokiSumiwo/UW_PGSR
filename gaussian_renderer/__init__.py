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
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from scene.medium_model import MediumEstimator
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           app_model: AppModel=None, medium_model: MediumEstimator=None, complete_physics_model=None, return_plane = True, return_depth_normal = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if not return_plane:
        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        # In the non-geometry path, treat rendered_image as air (clean) base if medium_model is provided
        air_image = rendered_image
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            air_image = torch.exp(appear_ab[0]) * air_image + appear_ab[1]

        if medium_model is not None:
            # No plane_depth available in this branch; do not apply medium without geometry
            underwater_image = air_image
        else:
            underwater_image = air_image

        return_dict =  {"render": underwater_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        return_dict.update({"air_image": torch.clamp(air_image, 0.0, 1.0)})
        return return_dict

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]

    # Pixel-wise hit map (H,W): "is there a geometry hit for this pixel?"
    # This is useful for background masking/losses (Innovation 3). Do NOT confuse with out_observe,
    # which is per-Gaussian (N,) and used with visibility_filter.
    depth_planar_2d = plane_depth if plane_depth.dim() == 2 else plane_depth.squeeze()
    hit_map = (depth_planar_2d > 0).to(depth_planar_2d.dtype)
    
    # Complete underwater physics rendering
    # Step 1: 3DGS renders clear scene (no water effects)
    clear_scene = rendered_image
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        clear_scene = torch.exp(appear_ab[0]) * clear_scene + appear_ab[1]
    
    # Step 2: Apply complete underwater physics
    if complete_physics_model is not None:
        # Use complete physics model for underwater effects (like 3D-UIR)
        # IMPORTANT: Use metric depth/path length z (meters) as physics input.
        # plane_depth is (H, W) from rasterizer; keep it as linear depth (do NOT invert).
        depth_planar = plane_depth if plane_depth.dim() == 2 else plane_depth.squeeze()
        # Use alpha to handle invalid/background pixels (plane_depth can be 0 where nothing is hit)
        # NOTE: we keep backward-compatible defaults via PipelineParams:
        #   - pipe.physics_close_background=False (default): use legacy behavior.
        #   - pipe.physics_alpha_grad=False (default): detach alpha (legacy).
        alpha_raw_2d = rendered_alpha.squeeze(0)  # (H,W)
        alpha_2d = alpha_raw_2d if getattr(pipe, "physics_alpha_grad", False) else alpha_raw_2d.detach()

        physics_max_depth = float(getattr(pipe, "physics_max_depth", 100.0))
        physics_bg_depth = float(getattr(pipe, "physics_bg_depth", 100.0))
        physics_alpha_thr = float(getattr(pipe, "physics_alpha_thr", 1e-4))

        if getattr(pipe, "physics_close_background", False):
            # --- Change 1: close "infinite / no-hit" pixels to B_inf, so UIFM handles far background ---
            # Use a hit-mask to avoid haloing (do not mix B_inf into soft edges unless actually hit).
            hit = (alpha_2d > physics_alpha_thr)
            alpha_mix = torch.where(hit, alpha_2d, torch.zeros_like(alpha_2d))

            # --- Scheme 4 (optional): depth-gated background closing ---
            # Even if alpha>0 due to residual far Gaussians, for sufficiently large depth we force alpha_eff -> 0,
            # so far contributions always return to B_inf and background stays consistent across novel views.
            if getattr(pipe, "physics_depth_gated_background", False) and bool(getattr(pipe, "physics_far_to_binf", 1)):
                # Optional: auto-quantile z_cut from hit-pixel depth distribution (per-view).
                z_cut = float(getattr(pipe, "bg_z_cut", 60.0))
                z_view = None
                if bool(getattr(pipe, "bg_z_cut_auto", 0)):
                    try:
                        with torch.no_grad():
                            q = float(getattr(pipe, "bg_z_cut_q", 0.95))
                            q = max(0.5, min(0.9999, q))
                            z_min = float(getattr(pipe, "bg_z_cut_min", 10.0))
                            z_max = float(getattr(pipe, "bg_z_cut_max", 200.0))
                            z_fb = float(getattr(pipe, "bg_z_cut_fallback", z_cut))
                            # Use hit pixels: depth>0 and alpha above threshold (same as "hit")
                            valid = (depth_planar > 0) & hit
                            if valid.any():
                                zq = float(torch.quantile(depth_planar[valid].detach().float(), q).item())
                                z_view = max(z_min, min(z_max, zq))
                            else:
                                z_view = z_fb
                    except Exception:
                        pass
                # 4.8.2: stabilize auto z_cut with EMA (optional). If beta<=0, fall back to per-view z_view.
                if z_view is not None:
                    beta = float(getattr(pipe, "bg_z_cut_ema_beta", 0.0))
                    if beta > 0.0:
                        try:
                            if not hasattr(pipe, "_bg_z_cut_ema"):
                                pipe._bg_z_cut_ema = float(z_view)
                            else:
                                pipe._bg_z_cut_ema = (1.0 - beta) * float(pipe._bg_z_cut_ema) + beta * float(z_view)
                            z_cut = float(pipe._bg_z_cut_ema)
                        except Exception:
                            z_cut = float(z_view)
                    else:
                        z_cut = float(z_view)
                z_tau = float(getattr(pipe, "bg_z_tau", 5.0))
                z_tau = max(z_tau, 1e-6)
                # Use planar depth (in meters) for gating; background pixels have alpha_mix=0 anyway.
                far_w = torch.sigmoid((depth_planar - z_cut) / z_tau)  # (H,W), far->1
                # Protect far-but-strong geometry from being suppressed (optional).
                if bool(getattr(pipe, "scheme4_far_protect", 1)):
                    a_prot = float(getattr(pipe, "scheme4_alpha_prot", 0.28))
                    a_tau = float(getattr(pipe, "scheme4_alpha_tau", 0.08))
                    a_tau = max(a_tau, 1e-6)
                    # p_alpha ~ 0 for thin fog alpha, ~1 for strong coverage alpha
                    p_alpha = torch.sigmoid((alpha_mix - a_prot) / a_tau).clamp(0.0, 1.0)
                    gate = (1.0 - far_w * (1.0 - p_alpha)).clamp(0.0, 1.0)
                    alpha_mix = torch.clamp(alpha_mix * gate, 0.0, 1.0)
                else:
                    alpha_mix = torch.clamp(alpha_mix * (1.0 - far_w), 0.0, 1.0)

            z_fg = torch.clamp(depth_planar, min=1e-3, max=physics_max_depth)
            depth_metric = torch.where(hit, z_fg, torch.full_like(z_fg, physics_bg_depth))

            # Inverse-depth feature: only meaningful for hit pixels; keep background stable as 0
            depth_inv = (1.0 / (depth_metric + 1e-6)) * hit.float()
            depth_inv = torch.clamp(depth_inv, min=0.0, max=2.0)
        else:
            # --- Legacy behavior (default): clamp depth to 100m and set background to 100m ---
            depth_metric = torch.clamp(depth_planar, min=1e-3, max=physics_max_depth)
            depth_metric = torch.where(alpha_2d > 0.0, depth_metric, torch.full_like(depth_metric, physics_bg_depth))

            depth_inv = (1.0 / (depth_metric + 1e-6)) * alpha_2d
            depth_inv = torch.clamp(depth_inv, min=0.0, max=2.0)
        
        # Generate camera info for appearance modeling
        # NOTE: Avoid guessing world_view_transform layout; use camera_center and R directly.
        cam_center = viewpoint_camera.camera_center
        cam_center_t = cam_center if torch.is_tensor(cam_center) else torch.tensor(cam_center, device=depth_metric.device)
        cam_center_t = cam_center_t.to(device=depth_metric.device, dtype=depth_metric.dtype).flatten()
        R_t = torch.tensor(viewpoint_camera.R, device=depth_metric.device, dtype=depth_metric.dtype).flatten()
        fov_t = torch.tensor([viewpoint_camera.FoVx, viewpoint_camera.FoVy], device=depth_metric.device, dtype=depth_metric.dtype)
        camera_info = torch.cat([cam_center_t, R_t, fov_t])
        
        # Ensure camera_info is 1D tensor
        if camera_info.dim() > 1:
            camera_info = camera_info.flatten()
        
        # Apply complete underwater physics: I = J * A(z) + B(z)
        depth_scale = getattr(pipe, "physics_depth_scale", 10.0)
        if getattr(pipe, "physics_close_background", False):
            # Explicitly compute physics components so we can close background to B_inf.
            backscatter, attenuation, appearance_embed = complete_physics_model(
                depth_metric, camera_info,
                depth_feature=depth_inv, depth_scale=depth_scale
            )
            I_fg = torch.clamp(clear_scene * attenuation + backscatter, 0.0, 1.0)
            B_inf = complete_physics_model.backscatter_net.B_inf  # (3,1,1)
            I_bg = B_inf.expand_as(I_fg)

            # Use alpha_mix if defined (hit-mask mode), else fall back to alpha_2d (should not happen)
            alpha3 = (alpha_mix if "alpha_mix" in locals() else alpha_2d).unsqueeze(0)
            underwater_image = torch.clamp(I_fg * alpha3 + I_bg * (1.0 - alpha3), 0.0, 1.0)
        else:
            underwater_image = complete_physics_model.apply_physics(
                clear_scene, depth_metric, camera_info,
                depth_feature=depth_inv, depth_scale=depth_scale
            )

            # Get physics components for analysis
            backscatter, attenuation, appearance_embed = complete_physics_model(
                depth_metric, camera_info,
                depth_feature=depth_inv, depth_scale=depth_scale
            )
        
        # Legacy compatibility
        a_map, B_map = attenuation, backscatter
        
    elif medium_model is not None:
        # Fallback to original medium model
        depth_planar = plane_depth if plane_depth.dim() == 2 else plane_depth.squeeze()
        a_map, B_map = medium_model(depth_planar)
        underwater_image = torch.clamp(a_map * clear_scene + B_map, 0.0, 1.0)
        backscatter, attenuation = B_map, a_map
        appearance_embed = None
        
    else:
        # No underwater effects
        a_map, B_map = None, None
        underwater_image = clear_scene
        backscatter, attenuation = None, None
        appearance_embed = None

    return_dict =  {"render": underwater_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "rendered_alpha": rendered_alpha,
                    "plane_depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    "hit_map": hit_map
                    }

    # Expose Scheme4 z_cut actually used in this render (useful for Scheme5/Innovation3 to stay consistent).
    # If Scheme4 is disabled, this falls back to the configured bg_z_cut.
    try:
        scheme4_z_cut_val = float(getattr(pipe, "bg_z_cut", 60.0))
        if getattr(pipe, "physics_close_background", False) and getattr(pipe, "physics_depth_gated_background", False) and bool(getattr(pipe, "physics_far_to_binf", 1)):
            if "z_cut" in locals():
                scheme4_z_cut_val = float(z_cut)
        return_dict["scheme4_z_cut"] = torch.tensor(scheme4_z_cut_val, device=underwater_image.device, dtype=underwater_image.dtype)
    except Exception:
        pass
    
    # Add clear scene (3DGS output without water effects)
    return_dict.update({"air_image": torch.clamp(clear_scene, 0.0, 1.0)})
    # Add the metric depth actually used by the physics model (for consistent losses/constraints)
    if complete_physics_model is not None:
        return_dict.update({"physics_depth": depth_metric, "physics_inv_depth": depth_inv})
    
    # Add physics components
    if a_map is not None:
        return_dict.update({
            "a_map": a_map, 
            "B_map": B_map, 
            "underwater_image": underwater_image,
            "backscatter": backscatter,
            "attenuation": attenuation,
            "appearance_embed": appearance_embed
        })

    # Expose the exact closed-rendering intermediates used above. These keys are
    # read-only diagnostics for visualization/export and do not alter rendering.
    if complete_physics_model is not None and getattr(pipe, "physics_close_background", False):
        if "alpha_mix" in locals():
            return_dict["alpha_mix"] = alpha_mix.unsqueeze(0)
        if "I_fg" in locals():
            return_dict["foreground_image"] = I_fg
        if "B_inf" in locals():
            return_dict["B_inf"] = B_inf
        if "hit" in locals():
            return_dict["physics_hit_mask"] = hit.unsqueeze(0).to(underwater_image.dtype)
        if "far_w" in locals():
            return_dict["far_gate"] = far_w.unsqueeze(0)
        if "p_alpha" in locals():
            return_dict["support_confidence"] = p_alpha.unsqueeze(0)
        if "gate" in locals():
            return_dict["alpha_mix_gate"] = gate.unsqueeze(0)

    if return_depth_normal:
        # render_normal expects 2D depth (H, W)
        depth_planar = plane_depth if plane_depth.dim() == 2 else plane_depth.squeeze()
        depth_normal = render_normal(viewpoint_camera, depth_planar) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict
