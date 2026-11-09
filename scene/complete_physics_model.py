import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_underwater_parameters(rgb_color, device='cuda'):
    """
    Initialize underwater physics parameters based on RGB color.
    
    Args:
        rgb_color: Tuple of (R, G, B) values in range [0, 255]
        device: Device to place tensors on
        
    Returns:
        dict: Dictionary containing initialized parameters
    """
    # Convert RGB to normalized values
    r, g, b = rgb_color
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    # Create parameter dictionary
    params = {
        'B_inf': torch.tensor([r_norm, g_norm, b_norm], device=device).view(3, 1, 1),
        'J_prime': torch.tensor([r_norm * 0.5, g_norm * 0.5, b_norm * 0.5], device=device).view(3, 1, 1),
        'b_infinity': torch.tensor([r_norm, g_norm, b_norm], device=device)
    }
    
    return params


class BackscatterNet(nn.Module):
    """
    Complete backscatter modeling network based on standard underwater imaging.
    Models: B(z) = B∞•(1 – e^(-βBz))
    """
    def __init__(self, underwater_color=(0, 105, 160), max_depth: float = 100.0):
        super().__init__()
        # For backward compatibility, keep the historical 100m clamp by default.
        # Set max_depth <= 0 to disable max clamp (only clamp min for numerical stability).
        self.max_depth = float(max_depth)
        # Convolutional layers for depth processing (like 3D-UIR)
        self.backscatter_conv = nn.Conv2d(1, 3, 1, bias=False)  # βB coefficients
        self.residual_conv = nn.Conv2d(1, 3, 1, bias=False)    # βD coefficients
        
        # Initialize weights
        nn.init.uniform_(self.backscatter_conv.weight, 0, 5.0)
        nn.init.uniform_(self.residual_conv.weight, 0, 5.0)
        
        # Initialize parameters based on underwater color
        params = initialize_underwater_parameters(underwater_color)
        self.B_inf = nn.Parameter(params['B_inf'])  # Underwater background color
        self.J_prime = nn.Parameter(params['J_prime'])  # Residual term with underwater color bias
        
        # Activation functions
        self.softplus = nn.Softplus()
        
    def forward(self, depth_metric, depth_feature=None, depth_scale: float = 10.0):
        """
        Compute backscatter component B(z)
        
        Args:
            depth_metric: Metric depth map z in meters (H, W) or (1, 1, H, W)
            depth_feature: Optional feature depth for coefficient prediction (e.g., inv depth 1/z)
            depth_scale: Scale factor to normalize metric depth (z_norm = z / depth_scale)
            
        Returns:
            backscatter: Backscatter component (3, H, W)
        """
        # Prepare metric depth tensor
        depth = depth_metric
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0)  # (1, 1, H, W)

        # Prepare feature depth (default: use metric depth as feature if not provided)
        feat = depth_feature
        if feat is None:
            feat = depth
        else:
            if feat.dim() == 2:
                feat = feat.unsqueeze(0).unsqueeze(0)
            elif feat.dim() == 3:
                feat = feat.unsqueeze(0)
            
        # Compute backscatter coefficients (like 3D-UIR)
        beta_b_conv = self.softplus(self.backscatter_conv(feat))  # βB (predicted from stable feature)
        beta_d_conv = self.softplus(self.residual_conv(feat))      # βD
        
        # Complete backscatter formula: B(z) = B∞•(1 – e^(-βBz)) + J'•e^(-βDz) (like 3D-UIR)
        # 数值稳定性改进
        if self.max_depth > 0:
            depth_clamped = torch.clamp(depth, min=1e-6, max=self.max_depth)
        else:
            depth_clamped = torch.clamp(depth, min=1e-6)
        # Normalize metric depth to reduce early saturation
        depth_norm = depth_clamped / max(float(depth_scale), 1e-6)
        
        # 第一项：B∞•(1 – e^(-βBz))
        exp_b = -beta_b_conv * depth_norm
        exp_b = torch.clamp(exp_b, min=-10.0, max=10.0)  # 限制指数范围
        term1 = self.B_inf * (1 - torch.exp(exp_b))
        
        # 第二项：J'•e^(-βDz)
        exp_d = -beta_d_conv * depth_norm
        exp_d = torch.clamp(exp_d, min=-10.0, max=10.0)  # 限制指数范围
        term2 = self.J_prime * torch.exp(exp_d)
        
        Bc = term1 + term2
        
        return Bc.squeeze(0)  # (3, H, W)


class AttenuationNet(nn.Module):
    """
    Complete attenuation modeling network based on 3D-UIR architecture.
    Models: A(z) = e^(-βDz)
    """
    def __init__(self, appearance_dim=16, max_depth: float = 100.0):
        super().__init__()
        self.appearance_dim = appearance_dim
        # For backward compatibility, keep the historical 100m clamp by default.
        # Set max_depth <= 0 to disable max clamp (only clamp min for numerical stability).
        self.max_depth = float(max_depth)
        
        # Convolutional layer for depth + appearance processing
        self.attenuation_conv = nn.Conv2d(1 + appearance_dim, 6, 1, bias=False)
        nn.init.uniform_(self.attenuation_conv.weight, 0, 2.0)  # 降低初始权重范围
        
        # Attenuation coefficients with better initialization
        self.attenuation_coef = nn.Parameter(torch.tensor([0.5, 0.3, 0.4, 0.2, 0.6, 0.1]).view(6, 1, 1))
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        
    def forward(self, depth_metric, appearance_embed=None, depth_feature=None, depth_scale: float = 10.0):
        """
        Compute attenuation component A(z)
        
        Args:
            depth_metric: Metric depth map z in meters (H, W) or (1, 1, H, W)
            appearance_embed: Appearance embedding (appearance_dim,)
            depth_feature: Optional feature depth for coefficient prediction (e.g., inv depth 1/z)
            depth_scale: Scale factor to normalize metric depth (z_norm = z / depth_scale)
            
        Returns:
            attenuation: Attenuation component (3, H, W)
        """
        depth = depth_metric
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0)

        feat = depth_feature
        if feat is None:
            feat = depth
        else:
            if feat.dim() == 2:
                feat = feat.unsqueeze(0).unsqueeze(0)
            elif feat.dim() == 3:
                feat = feat.unsqueeze(0)
            
        # Prepare appearance embedding
        if appearance_embed is not None:
            if appearance_embed.dim() == 1:
                appearance_embed = appearance_embed.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, dim, 1, 1)
            appearance_embed = appearance_embed.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
            depth_embed = torch.cat((feat, appearance_embed), dim=1)
        else:
            # Use zero embedding if not provided
            zero_embed = torch.zeros(feat.shape[0], self.appearance_dim, feat.shape[2], feat.shape[3], 
                                   device=feat.device, dtype=feat.dtype)
            depth_embed = torch.cat((feat, zero_embed), dim=1)
        
        # Compute attenuation coefficients
        attn_conv = torch.exp(-self.sigmoid(self.attenuation_conv(depth_embed)))
        
        # Combine coefficients for RGB channels
        beta_d = torch.stack(tuple(
            torch.sum(attn_conv[:, i:i + 2, :, :] * self.sigmoid(self.attenuation_coef[i:i + 2]), dim=1) 
            for i in range(0, 6, 2)), dim=1)
        
        # Attenuation formula: A(z) = e^(-βDz)
        # 改进数值稳定性，避免过大的指数值
        beta_d_clamped = torch.clamp(beta_d, 0.0, 5.0)  # 进一步限制最大衰减系数
        if self.max_depth > 0:
            depth_clamped = torch.clamp(depth, min=1e-6, max=self.max_depth)
        else:
            depth_clamped = torch.clamp(depth, min=1e-6)
        depth_norm = depth_clamped / max(float(depth_scale), 1e-6)
        
        # 数值稳定的指数计算
        exponent = -beta_d_clamped * depth_norm
        exponent = torch.clamp(exponent, min=-10.0, max=10.0)  # 限制指数范围
        attenuation_map = torch.exp(exponent)
        
        # 移除sigmoid，保持物理意义的指数衰减
        # attenuation_map = self.sigmoid(attenuation_map)  # 注释掉，保持物理意义
        
        return attenuation_map.squeeze(0)  # (3, H, W)


class AppearanceNet(nn.Module):
    """
    Appearance modeling network for view-dependent effects.
    """
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, camera_info):
        """
        Generate appearance embedding
        
        Args:
            camera_info: Camera information (position, orientation, etc.)
            
        Returns:
            appearance_embed: Appearance embedding (output_dim,)
        """
        return self.network(camera_info)


class CompleteUnderwaterPhysics(nn.Module):
    """
    Complete underwater physics model combining backscatter and attenuation.
    Implements the full underwater imaging formula: I = J * A(z) + B(z)
    """
    def __init__(self, appearance_dim=16, underwater_color=(0, 105, 160), max_depth: float = 100.0):
        super().__init__()
        # Keep backward-compatible 100m clamp by default (see BackscatterNet/AttenuationNet).
        self.max_depth = float(max_depth)
        self.backscatter_net = BackscatterNet(underwater_color, max_depth=self.max_depth)
        self.attenuation_net = AttenuationNet(appearance_dim, max_depth=self.max_depth)
        self.appearance_net = AppearanceNet(output_dim=appearance_dim)
        
        # Optimizer with learning rate 默认与 PipelineParams.physics_model_lr 对齐（0.0009）
        self.optimizer = torch.optim.Adam(self.parameters(), lr=9e-4, betas=(0.9, 0.99))
        
    def forward(self, depth_metric, camera_info=None, depth_feature=None, depth_scale: float = 10.0):
        """
        Complete underwater physics forward pass
        
        Args:
            depth_metric: Metric depth map z (H, W) in meters
            camera_info: Camera information for appearance modeling
            depth_feature: Optional feature depth for coefficient prediction (e.g., inv depth 1/z)
            depth_scale: Scale factor to normalize metric depth in exponential terms
            
        Returns:
            backscatter: Backscatter component (3, H, W)
            attenuation: Attenuation component (3, H, W)
            appearance_embed: Appearance embedding (appearance_dim,)
        """
        # 数值稳定性检查
        if self.max_depth > 0:
            depth = torch.clamp(depth_metric, min=1e-6, max=self.max_depth)
        else:
            depth = torch.clamp(depth_metric, min=1e-6)
        
        # Generate appearance embedding
        if camera_info is not None:
            appearance_embed = self.appearance_net(camera_info)
        else:
            appearance_embed = torch.zeros(self.appearance_net.network[-1].out_features, device=depth.device, dtype=depth.dtype)
        
        # Compute physics components with numerical stability
        backscatter = self.backscatter_net(depth, depth_feature=depth_feature, depth_scale=depth_scale)
        attenuation = self.attenuation_net(depth, appearance_embed, depth_feature=depth_feature, depth_scale=depth_scale)
        
        # 数值稳定性检查
        backscatter = torch.clamp(backscatter, min=0.0, max=1.0)
        attenuation = torch.clamp(attenuation, min=0.0, max=1.0)
        
        # 检查NaN值
        if torch.isnan(backscatter).any():
            print("Warning: NaN detected in backscatter, replacing with zeros")
            backscatter = torch.zeros_like(backscatter)
        if torch.isnan(attenuation).any():
            print("Warning: NaN detected in attenuation, replacing with ones")
            attenuation = torch.ones_like(attenuation)
        
        return backscatter, attenuation, appearance_embed
    
    def apply_physics(self, clear_scene, depth_metric, camera_info=None, depth_feature=None, depth_scale: float = 10.0):
        """
        Apply underwater physics to clear scene
        
        Args:
            clear_scene: Clear scene image (3, H, W)
            depth: Depth map (H, W)
            camera_info: Camera information
            
        Returns:
            underwater_image: Underwater image (3, H, W)
        """
        backscatter, attenuation, _ = self.forward(depth_metric, camera_info, depth_feature=depth_feature, depth_scale=depth_scale)
        
        # Underwater imaging formula: I = J * A(z) + B(z)
        underwater_image = clear_scene * attenuation + backscatter
        
        return underwater_image.clamp(0.0, 1.0)
    
    @staticmethod
    def tv_loss(x: torch.Tensor):
        """Total variation loss for regularization"""
        dh = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean()
        dw = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
        return dh + dw
    
    def physics_constraints(self, backscatter, attenuation, depth):
        """
        Physics-based constraints for underwater imaging
        
        Args:
            backscatter: Backscatter component (3, H, W)
            attenuation: Attenuation component (3, H, W)
            depth: Metric depth / path length z in meters (H, W)
            
        Returns:
            constraint_loss: Physics constraint loss
        """
        # NOTE: depth is metric z (meters), consistent with the forward physics model.
        # Constraint 1: Attenuation should decrease with depth (monotonic)
        # 当深度增加时，衰减应该减小（A(z) = e^(-βz) 单调递减）
        if depth.shape[0] > 1 and depth.shape[1] > 1:
            # 计算倒数深度梯度（带方向）
            depth_grad_x = depth[1:, :] - depth[:-1, :]
            depth_grad_y = depth[:, 1:] - depth[:, :-1]
            
            # 计算衰减梯度（带方向）
            attn_grad_x = attenuation[:, 1:, :] - attenuation[:, :-1, :]
            attn_grad_y = attenuation[:, :, 1:] - attenuation[:, :, :-1]
            
            # 单调性约束：倒数深度增加时衰减应该减小
            # 即：depth_grad > 0 时，attn_grad < 0
            # 违反单调性的情况：attn_grad * depth_grad > 0
            attn_violation_x = torch.clamp(attn_grad_x * depth_grad_x, min=0.0)
            attn_violation_y = torch.clamp(attn_grad_y * depth_grad_y, min=0.0)
            monotonic_loss = torch.mean(attn_violation_x) + torch.mean(attn_violation_y)
        else:
            monotonic_loss = torch.tensor(0.0, device=attenuation.device)
        
        # Constraint 2: Backscatter should increase with depth
        # 后向散射应该随深度增加而增加
        if depth.shape[0] > 1 and depth.shape[1] > 1:
            # 计算倒数深度梯度（带方向）
            depth_grad_x = depth[1:, :] - depth[:-1, :]
            depth_grad_y = depth[:, 1:] - depth[:, :-1]
            
            # 计算后向散射梯度（带方向）
            backscatter_grad_x = backscatter[:, 1:, :] - backscatter[:, :-1, :]
            backscatter_grad_y = backscatter[:, :, 1:] - backscatter[:, :, :-1]
            
            # 单调性约束：倒数深度增加时后向散射应该增加
            # 即：depth_grad > 0 时，backscatter_grad > 0
            # 违反单调性的情况：-backscatter_grad * depth_grad > 0
            bs_violation_x = torch.clamp(-backscatter_grad_x * depth_grad_x, min=0.0)
            bs_violation_y = torch.clamp(-backscatter_grad_y * depth_grad_y, min=0.0)
            backscatter_monotonic = torch.mean(bs_violation_x) + torch.mean(bs_violation_y)
        else:
            backscatter_monotonic = torch.tensor(0.0, device=attenuation.device)
        
        # Constraint 3: Physical bounds (0 ≤ A(z) ≤ 1, 0 ≤ B(z) ≤ 1)
        attn_bounds_loss = torch.mean(torch.clamp(attenuation, 0.0, 1.0) - attenuation)
        backscatter_bounds_loss = torch.mean(torch.clamp(backscatter, 0.0, 1.0) - backscatter)
        bounds_loss = attn_bounds_loss + backscatter_bounds_loss
        
        # Constraint 4: 在无限深度处，A(z) → 0, B(z) → B∞
        # 这个约束通过指数函数自然满足
        
        total_constraint = 0.1 * monotonic_loss + 0.1 * backscatter_monotonic + 0.1 * bounds_loss
        return total_constraint
    
    def save_weights(self, model_path: str, iteration: int):
        """Save model weights"""
        import os
        out_weights_path = os.path.join(model_path, "complete_physics/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        print(f"Save complete physics model. path: {out_weights_path}")
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'physics.pth'))
    
    def load_weights(self, model_path: str, iteration: int = -1):
        """Load model weights"""
        import os
        def _searchForMaxIteration(folder):
            try:
                saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
                return max(saved_iters)
            except Exception:
                return None
                
        if iteration == -1:
            folder = os.path.join(model_path, "complete_physics")
            loaded_iter = _searchForMaxIteration(folder)
            if loaded_iter is None:
                raise FileNotFoundError("No saved complete_physics weights found at {}".format(folder))
        else:
            loaded_iter = iteration
            
        weights_path = os.path.join(model_path, "complete_physics/iteration_{}/physics.pth".format(loaded_iter))
        state_dict = torch.load(weights_path, map_location='cuda')
        self.load_state_dict(state_dict)
