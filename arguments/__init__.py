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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._air_source_path = ""
        # Optional: pseudo "air/clear-scene" supervision root (e.g. Deep_Sea-NN outputs).
        # If empty, we will auto-detect under: <source_path>/<images_root>/<fake_air_folder_name>/
        # Default expected layout from user:
        #   source_path/images/fake_air_Deep_Sea-NN/<same filename as training image>
        self._fake_air_source_path = ""
        self.fake_air_folder_name = "fake_air_Deep_Sea-NN"
        self._model_path = ""
        # Default to images/ColorImage per user's dataset layout
        self._images = "images/ColorImage"
        self._depths = "depths"
        # Keep original resolution by default (no auto downscale)
        self._resolution = 1
        self._white_background = False
        self.data_device = "cuda"
        # Default to evaluation mode (matches exp22 command line)
        self.eval = True
        self.preload_img = True
        self.ncc_scale = 1.0
        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 1.5
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        # Medium model switches
        self.use_underwater_model = True
        self.medium_num_basis = 4
        self.medium_tv_weight = 0.01
        # --- Complete physics renderer controls (background closing / depth handling) ---
        # NOTE: Defaults below are set to the user's "best" training command (enabled by default).
        # Use 0/1 for toggles so you can still disable them from CLI/config (0=off, 1=on).
        # When enabled, renderer will close no-hit background pixels to B_inf:
        #   I = alpha*(J*A+B) + (1-alpha)*B_inf
        self.physics_close_background = 1
        # Alpha threshold to define "hit" pixels in background closing mode.
        self.physics_alpha_thr = 1e-4
        # Maximum depth (meters) used for foreground clamp. Legacy default was 100.
        self.physics_max_depth = 1_000_000.0
        # Background depth (meters) used for no-hit pixels. In legacy mode this was 100.
        # In background-closing mode you can set it very large (e.g. 1e6) to mimic infinity.
        self.physics_bg_depth = 1_000_000.0
        # Whether to keep gradient through alpha in physics rendering branch.
        # Default False keeps legacy detach behavior; set True to let background error push alpha->0.
        self.physics_alpha_grad = 1
        # Physics network internal depth clamp (meters).
        # Default 100 keeps historical behavior; set very large (e.g. 1000/1e6) or <=0 to disable max clamp.
        self.physics_net_max_depth = 1_000_000.0

        # --- Scheme 4/5 knobs (default OFF / 0.0 for backward compatibility) ---
        # Scheme 4: depth-gated background closing (alpha_eff = alpha * (1-sigmoid((z-z_cut)/tau)))
        self.physics_depth_gated_background = 1
        # NEW (separate switch): when depth-gated background is enabled, apply the far-depth gate to the
        # *underwater render* (i.e. drive far-hit pixels back to B_inf by reducing alpha_mix).
        # Keep this ON by default. Turning it OFF keeps "no-hit -> B_inf" closure intact, but does NOT
        # force far-hit pixels to B_inf.
        self.physics_far_to_binf = 1
        # NEW (recommended): protect far *real geometry* from being suppressed by Scheme4.
        # Only suppress far pixels if they are also "weak hits". A robust first signal is alpha strength:
        #   p_alpha = sigmoid((alpha - alpha_prot)/alpha_tau)
        #   alpha_eff = alpha * (1 - far_w * (1 - p_alpha))
        # so far-but-strong alpha stays, far-but-thin fog gets removed.
        # Default ON with conservative thresholds; turn OFF to get the original "depth-only" gating.
        self.scheme4_far_protect = 1
        self.scheme4_alpha_prot = 0.28   # e.g. 0.20~0.35
        self.scheme4_alpha_tau = 0.08    # e.g. 0.05~0.12
        self.bg_z_cut = 60.0   # (meters) beyond this, force alpha_eff -> 0 (treat as infinity background)
        self.bg_z_tau = 5.0    # (meters) transition width
        # Optional: auto-quantile z_cut (per-view) computed from hit-pixel depths.
        # Default ON (recommended): auto z_cut with EMA stabilizer.
        self.bg_z_cut_auto = 1  # 0/1
        self.bg_z_cut_q = 0.95
        self.bg_z_cut_min = 10.0
        self.bg_z_cut_max = 200.0
        self.bg_z_cut_fallback = 60.0
        # Auto z_cut stabilizer (EMA across views/iters). Only used when bg_z_cut_auto==1.
        # If <=0, fall back to per-view quantile (legacy auto behavior).
        self.bg_z_cut_ema_beta = 0.05
        # Scheme 5: far-alpha regularization weight (set to >0 to enable)
        # Default OFF (C-best): bg_hit + bg_prune + densify_block_far are sufficient and safer.
        self.lambda_far_alpha = 0.0
        # NEW: Scheme5 protection switch (decouple from Scheme4).
        # If 1: penalize only far&weak alpha using Scheme4-like p_alpha protection.
        # If 0: penalize all far alpha (no protection). Useful for ablations.
        self.scheme5_far_protect = 1
        # Only use depth as a weight (detach) in far-alpha regularizer to avoid pushing geometry.
        self.far_alpha_detach_depth = 1
        # Scheme 5 schedule (optional): ramp lambda_far_alpha from 0 -> lambda_far_alpha
        # Default 0 disables ramp (legacy behavior).
        self.far_alpha_from_iter = 0
        self.far_alpha_ramp_iters = 0

        # --- Innovation 1: block far densify (capacity routing), default OFF ---
        # When enabled (after warmup), exclude far Gaussians from densify stats so they are unlikely to clone/split.
        self.densify_block_far = 1
        # Start iteration to enable densify blocking (warmup). If <=0, starts immediately when densify starts.
        self.densify_block_far_from_iter = 1000
        # Optional: ramp-in densify blocking (loose->tight) after densify_block_far_from_iter.
        # If >0, effective z_cut will interpolate from densify_z_cut_max -> computed z_cut.
        self.densify_block_far_ramp_iters = 3000
        # Adaptive z_cut (meters) from depth quantile over hit pixels (per-view)
        self.densify_z_cut_q = 0.90
        self.densify_z_cut_min = 10.0
        self.densify_z_cut_max = 80.0
        self.densify_z_cut_fallback = 20.0
        # Alpha threshold to define hit pixels when computing z_cut
        self.densify_hit_alpha_thr = 1e-4

        # --- Eval-time option (render.py): black out air-image background with the SAME gating logic as Scheme 4 ---
        # Default OFF. Uses alpha_eff = alpha_hit * (1 - sigmoid((z - bg_z_cut)/bg_z_tau)).
        # NOTE: Stored in PipelineParams so render.py can load it via get_combined_args/cfg_args.
        self.air_bg_black = 0  # 0/1
        # If >0, overrides physics_alpha_thr for air gating. If <=0, use physics_alpha_thr.
        self.air_bg_alpha_thr = -1.0
        # 0/1: apply depth gate using bg_z_cut/bg_z_tau (Scheme 4). If 0, just use alpha_hit.
        self.air_bg_use_depth_gate = 1
        # If >0, overrides bg_z_cut/bg_z_tau for air gating. If <=0, use bg_z_cut/bg_z_tau.
        self.air_bg_z_cut = -1.0
        self.air_bg_z_tau = -1.0
        # 0/1: additionally require pixel hit_map (plane_depth>0) besides alpha_thr.
        self.air_bg_use_hit_map = 0
        # If enabled, fill depth EXR of bg-black pixels as 1.0 (==1000m cap) for consistency.
        self.air_bg_depth_fill_one = 1  # 0/1 (only used when air_bg_black==1)
        self.air_bg_depth_one_thr = 1e-3  # alpha_eff<thr treated as bg-black for depth fill

        # --- Train-time option: gate air_image to black (for fake-air supervision consistency) ---
        # Default OFF. When enabled, air_image used in fake-air loss will be gated with the SAME alpha_eff
        # as Scheme 4 (depth-gated background closing), i.e. air_image *= alpha_eff.
        self.train_air_bg_black = 0  # 0/1
        # If >0, overrides physics_alpha_thr for train air gating. If <=0, use physics_alpha_thr.
        self.train_air_bg_alpha_thr = -1.0
        # If >0, overrides bg_z_cut/bg_z_tau for train air gating. If <=0, use bg_z_cut/bg_z_tau.
        self.train_air_bg_z_cut = -1.0
        self.train_air_bg_z_tau = -1.0

        # --- Innovation 2: per-Gaussian far-opacity penalty, default OFF (weight=0) ---
        # NOTE: Innovation 2 (far-opacity/far-scale) was removed by user request (keep only Innovation 1 + add Innovation 3).
        # Physics model hyperparameters (可调超参数，默认与 config.sh 中设置一致)
        # NOTE: For stability, we use inv-depth as feature but keep metric depth in formulas.
        # physics_depth_scale controls normalization of metric depth inside exp terms: z_norm = z / physics_depth_scale.
        # Defaults below are set to exp22 (best in your sweep).
        self.physics_depth_scale = 12.0        # (meters) depth normalization scale
        self.physics_model_lr = 0.00035        # exp22
        self.physics_tv_weight = 0.006         # exp22
        self.physics_constraint_weight = 0.03  # exp22
        # Gradient scaling and depth thresholds for physics-aware densification（对齐命令行）
        self.physics_region_depth_threshold = 2.6   # exp22
        self.water_region_depth_threshold = 2.0     # exp22
        self.physics_region_gradient_scale = 0.12   # exp22
        self.water_region_gradient_scale = 0.06     # exp22
        self.physics_strength_threshold = 0.32      # exp22
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0004 # 原0.0002，调大以抑制生长
        self.scale_loss_weight = 100.0

        # DRO parameters - 混合方案C: DRO剪枝代替DRO Loss
        # 关闭DRO Loss，避免与多视图冲突
        self.use_dro_loss = False              # 关闭DRO Loss
        self.lambda_depth = 0.0                # DRO Loss权重设为0（不使用）
        self.depth_loss_from_iter = 500        # 保留参数（兼容性）
        self.lambda_smooth = 0.0               # 关闭平滑Loss
        self.lambda_tv = 0.0                   # 关闭TV Loss
        self.tv_from_iter = 500
        self.tv_until_iter = 30000
        
        # DRO Guided Pruning parameters - 使用DRO深度图指导剪枝
        # 当前默认采用实验中表现最好的配置：
        #   --dro_guided_pruning
        #   --dro_prune_from_iter 6000
        #   --dro_prune_until_iter 16000
        #   --dro_prune_interval 50
        #   --max_dro_prune_per_iter 800
        #   --dro_depth_diff_threshold -0.3
        #   --dro_prune_max_depth 7.0
        #   --dro_prune_opacity_threshold 0.5
        # 原始默认值（仅作备忘，方便 ablation 时还原）：
        #   dro_prune_from_iter = 7000
        #   dro_prune_until_iter = 15000
        #   dro_prune_interval = 100
        #   max_dro_prune_per_iter = 400
        #   dro_depth_diff_threshold = -0.35
        #   dro_prune_max_depth = 5.0
        #   dro_prune_opacity_threshold = 0.4
        self.dro_guided_pruning = False        # 启用DRO指导剪枝
        self.dro_prune_from_iter = 6000        # 从6000次开始DRO剪枝
        self.dro_prune_until_iter = 16000      # 到16000次停止DRO剪枝
        self.dro_prune_interval = 50           # 每50次迭代执行一次DRO剪枝
        self.max_dro_prune_per_iter = 800      # 每次迭代最多剪枝800个点
        self.dro_depth_diff_threshold = -0.3   # 深度差异阈值（归一化，负值表示PGSR更近）
        self.dro_prune_max_depth = 7.0         # 只剪枝距离相机7米内的点
        self.dro_prune_opacity_threshold = 0.5 # 只剪枝不透明度<0.5的点
        
        # Depth-aware aggressive pruning parameters (inspired by 3D-UIR)
        self.far_prune_distance = 20.0        # 远距离剪枝阈值(米)
        self.far_opacity_threshold = 0.3      # 远距离不透明度阈值
        self.very_far_prune_distance = 50.0   # 极远距离强制剪枝(米)
        self.aggressive_far_prune = True      # 是否启用激进远距离剪枝
        self.aggressive_opacity_reset = True  # 是否使用3D-UIR风格的激进opacity重置
        self.reset_opacity_value = 0.01       # 重置时的opacity值 (3D-UIR uses very low value)
        
        self.wo_image_weight = False
        self.single_view_weight = 0.015
        self.single_view_weight_from_iter = 7000

        # Metric3D normal prior (optional, default OFF)
        self.use_metric_normal_prior = False   # 是否启用 Metric3D 法线先验
        self.lambda_metric_normal = 0.02       # 法线先验权重（保持较小，避免与原几何约束冲突）
        self.metric_normal_from_iter = 9000    # 从哪个迭代开始启用法线先验（晚于单视/多视）
        self.metric_conf_min = 0.0             # 置信度归一化下界
        self.metric_conf_max = 8.0             # 置信度归一化上界（裁剪过大值）

        # Default to using virtual cameras (matches exp22 command line)
        self.use_virtul_cam = True
        self.virtul_cam_prob = 0.1
        self.use_multi_view_trim = True
        self.multi_view_ncc_weight = 0.15        # 恢复原始权重（不再与DRO Loss冲突）
        self.multi_view_geo_weight = 0.03        # 恢复原始权重
        self.multi_view_weight_from_iter = 7000
        self.multi_view_weight_rampup_iter = 3000  # 渐进式激活：7k-10k线性增长到最终权重
        self.multi_view_patch_size = 3
        self.multi_view_sample_num = 102400
        self.multi_view_pixel_noise_th = 1.0
        self.wo_use_geo_occ_aware = False

        self.opacity_cull_threshold = 0.005
        self.densify_abs_grad_threshold = 0.0008
        self.abs_split_radii2D_threshold = 20
        self.max_abs_split_points = 50_000
        self.max_all_points = 6000_000
        self.exposure_compensation = False
        self.random_background = False
        # Appearance embedding switch (keep existing app_model behavior)
        self.use_appearance_embed = True

        # --- Innovation 3: background consistency loss (late-stage, robustly gated) ---
        # Default ON (bz works best in user's sweep). Set lambda_bg_consistency=0 to disable.
        self.lambda_bg_consistency = 0.002
        self.bg_loss_from_iter = 16000
        self.bg_loss_ramp_iters = 3000
        # Mask building hyperparams (safe defaults; tune per-scene)
        self.bg_border = 24
        self.bg_grad_thr = 0.02
        # v2: do not rely on fixed water color; use hit-map + far/tex/nohit score to pick seeds
        self.bg_use_hit_map = True           # prefer out_observe over alpha for hit/nohit
        self.bg_z_q = 0.90                   # depth quantile to define z_cut (over hit pixels)
        self.bg_z_tau_ratio = 0.05           # z_tau = max(z_tau_ratio*z_cut, 1e-3)
        self.bg_seed_top_q = 0.98            # pick seeds by top-quantile score
        self.bg_min_seed_ratio = 0.01
        self.bg_seed_std_thr = 0.05          # allow looser seed std; soft mask will down-weight uncertain pixels
        # adaptive color sigma (seed statistics) instead of fixed eps
        self.bg_sigma_col_min = 0.02
        self.bg_sigma_col_max = 0.12
        self.bg_morph_k = 5                  # soft mask smoothing kernel (avg pool)
        self.bg_min_pixels = 2048
        self.bg_alpha_weight = 0.2  # optional alpha suppression weight within bg mask
        # Target for bg color supervision:
        # - "binf": use physics B_inf if available (recommended)
        # - "bz":   use physics backscatter map B(z) if available
        # - "bgt":  use seed-estimated B_gt (legacy / fallback)
        # Default background target for Innovation 3
        self.bg_target_mode = "bz"
        # Optional weak calibration: pull physics target toward seed B_gt when seed is very clean
        self.bg_calib_weight = 0.0

        # --- Innovation 3 (v2): suppress "bg-hit" alpha (far&weak) in addition to no-hit color consistency ---
        # Default OFF to preserve exp_bg_bz_12 baseline. Enable in 4.8.2 experiments.
        # Innovation3 (bg_hit alpha suppression): default ON (C-best)
        self.bg_hit_alpha_weight = 0.04
        self.bg_hit_from_iter = 12000
        self.bg_hit_ramp_iters = 3000

        # --- New: Background double-evidence pruning (robust, default OFF) ---
        # NOTE (2025-12): pruning logic has been upgraded to a *single* BG-mask trigger aligned with
        # `tools/check_masks.py`: BG_MASK(GT + rendered_alpha + B_inf) ∩ (rendered_alpha > bg_prune_presence_thr).
        # The old "double evidence" knobs are kept for backward compatibility with cfg_args, but are deprecated.
        # Set default ON to match user's latest stable cfg_args (exp_bg_bz_12).
        self.bg_prune_enable = True
        self.bg_prune_from_iter = 12000
        # If >0, stop pruning after this iter. Default -1 means "no upper bound" (allow pruning until end of training).
        self.bg_prune_until_iter = -1
        self.bg_prune_interval = 100
        # 0/1: print skip reasons and intermediate counters for debugging pruning not firing.
        self.bg_prune_debug = 0
        # BG-mask threshold (soft -> bin)
        self.bg_prune_mask_thr = 0.25        # threshold on soft bg mask for intersection
        self.bg_prune_mask_ratio_min = 0.002 # skip if mask too tiny (fraction of pixels)
        self.bg_prune_mask_ratio_max = 0.40  # skip if mask too huge
        # Pixel-level "gaussian present" threshold inside bg region (alpha > thr)
        self.bg_prune_presence_thr = 1e-4
        # Map mask->gaussians: per-gaussian center sampling on a dilated pyramid
        self.bg_prune_cov_thr = 0.5          # sampled overlap score threshold
        # Safety gates (per-gaussian)
        self.bg_prune_edge_thr = 0.05        # protect strong GT edges
        self.bg_prune_opacity_thr = 0.6      # only prune relatively low-opacity (safer)
        # Deprecated (kept for old cfg_args): old double-evidence knobs (no longer used by default)
        self.bg_prune_far_w_thr = 0.6
        self.bg_prune_hit_alpha_thr = -1.0   # if <=0 use pipe.physics_alpha_thr
        self.bg_prune_z_min = -1.0           # if <=0 use pipe.bg_z_cut
        self.bg_prune_use_far = 1
        self.bg_prune_use_bg = 1
        # Extra heuristics: large and isolated floaters (0/1 for CLI toggles)
        self.bg_prune_use_size_gate = 1
        self.bg_prune_min_radii = 16.0       # pixels (per-view radii)
        self.bg_prune_use_isolation = 1
        self.bg_prune_iso_grid = 32          # pixels (cell size for coarse occupancy)
        self.bg_prune_iso_max_count = 2      # max points per cell to consider "isolated"
        # Accumulation across iterations (robustness)
        self.bg_prune_seen_min = 5
        self.bg_prune_ratio_thr = 0.7
        # Limit damage per pruning event
        self.bg_prune_max_per_iter = 800

        # --- NEW: BG mask (GT + rendered_alpha + B_inf) used by pruning & bg-consistency (aligns tools/check_masks.py) ---
        # Pixel-level "gaussian present" threshold inside background region.
        self.bg_prune_presence_thr = 1e-4
        # Alpha-low prior for background likelihood: w_alpha = ((thr - alpha)/thr)^gamma
        self.bgm_alpha_bg_thr = 0.10
        self.bgm_alpha_gamma = 1.5
        # Weight for GT/B_inf color term vs alpha-low term in bg likelihood (0..1). Higher -> more GT/B_inf.
        self.bgm_gt_w = 0.58
        # Seed/valid gating
        self.bgm_seed_gray_min = 0.03
        self.bgm_seed_alpha_thr = 0.03
        self.bgm_seed_binf_eps = 0.35
        self.bgm_min_seed_px = 512
        self.bgm_seed_std_thr = 0.08
        # Per-view calibration of B_inf towards seed median: B_ref=(1-w)*B_inf + w*B_seed
        self.bgm_binf_calib_w = 0.5
        # Optional hard gate on color closeness to B_ref
        self.bgm_use_binf_gate = 1
        self.bgm_binf_eps = 0.25
        # Soft color weight bandwidth and threshold
        self.bgm_sigma_col_min = 0.02
        self.bgm_sigma_col_max = 0.15
        self.bgm_col_thr = 0.35
        # Morphology smoothing kernel
        self.bgm_morph_k = 5

        # --- Console logging (tqdm) ---
        # Default False keeps legacy verbose tqdm postfix.
        # When True, show a short set of scalars for easier copy/paste:
        #   Loss/L1/SSIM/FakeAir/BGPrune/Points
        self.tqdm_short = False

        # === Optional: pseudo clear-scene (fake air) supervision on 3DGS "air_image" ===
        # When fake air images exist, we add a small L1 constraint:
        #   L += lambda_fake_air * || air_image - fake_air_gt ||_1
        # This helps disentanglement by anchoring the pure-3DGS rendering to "no-water" pseudo targets.
        # NOTE: we intentionally do NOT rely on a boolean flag here, because ParamGroup uses store_true
        # (no easy --no_xxx). Use lambda_fake_air=0 to disable.
        self.use_fake_air_supervision = False
        # Default ON (if fake-air images exist). This was empirically beneficial in your Bay ablations.
        self.lambda_fake_air = 0.02
        self.fake_air_loss_from_iter = 0

        # --- Stronger fake-air supervision (proven useful in ablation): Laplacian (high-frequency) ---
        # Add high-frequency term on air_image vs fake_air_gt:
        #   L += lambda_fake_air_lap * || Lap(air_image) - Lap(fake_air_gt) ||_1
        # Default ON (if fake-air images exist).
        self.lambda_fake_air_lap = 0.05         # e.g. 0.03~0.08
        # --- LPIPS (VGG) perceptual loss on fake-air (restore exp01 behavior) ---
        # NOTE: computed sparsely in train.py for speed. Default ON.
        self.lambda_fake_air_lpips = 0.01       # typical 0.005~0.02
        self.fake_air_lpips_interval = 20       # compute LPIPS every N iters
        self.fake_air_lpips_downsample = 2      # downsample factor for LPIPS (2 or 4)
        # Heuristic confidence (no uncertainty map available): suppress fake-air hallucinations
        self.fake_air_edge_ratio_thr = 2.0      # if fake-air edge >> underwater edge, down-weight
        self.fake_air_edge_ratio_k = 1.0        # strength of down-weighting
        self.fake_air_depth_tau = 12.0          # exp(-z/tau): down-weight far pixels in fake-air supervision
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
