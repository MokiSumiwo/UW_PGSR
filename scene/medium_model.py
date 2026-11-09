import torch
import torch.nn as nn
import torch.nn.functional as F


class MediumEstimator(nn.Module):
    """
    Enhanced underwater medium estimator based on real underwater imaging formula:
    I = J•e^(-βDz) + B∞•(1 – e^(-βBz))
    
    Where:
    - A(z) = e^(-βDz): direct attenuation (transmission)
    - B(z) = B∞•(1 – e^(-βBz)): backscatter component
    
    This model separates direct attenuation (βD) and backscatter attenuation (βB)
    with far-field scattered light intensity (B∞) for more accurate physics.
    """

    def __init__(self, num_basis: int = 6, use_spatial_smooth: bool = True, underwater_color=(0, 105, 160)):
        super().__init__()
        self.num_basis = num_basis
        self.use_spatial_smooth = use_spatial_smooth

        # Direct attenuation coefficients (βD) - for transmission
        beta_d_range = torch.logspace(start=-2, end=1.0, steps=num_basis)  # ~[0.01 .. 10]
        # Backscatter attenuation coefficients (βB) - for backscatter
        beta_b_range = torch.logspace(start=-2, end=1.0, steps=num_basis)  # ~[0.01 .. 10]
        
        self.register_buffer("beta_d_range", beta_d_range)
        self.register_buffer("beta_b_range", beta_b_range)

        # Direct attenuation (A(z)) parameters
        self.a_weights = nn.Parameter(torch.zeros(3, num_basis))
        self.a_bias = nn.Parameter(torch.zeros(3))

        # Backscatter (B(z)) parameters - including B∞
        self.b_weights = nn.Parameter(torch.zeros(3, num_basis))
        self.b_bias = nn.Parameter(torch.zeros(3))
        # Far-field scattered light intensity (B∞) - per channel
        # Initialize based on underwater color
        r, g, b = underwater_color
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        self.b_infinity = nn.Parameter(torch.tensor([r_norm, g_norm, b_norm]))  # Underwater background color

        if use_spatial_smooth:
            # Depth preprocessing: 3x3 conv + ReLU for a shallow feature
            self.depth_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        else:
            self.depth_conv = None

        # Optimizer will be created outside and set as attribute for convenience
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.99))

    def forward(self, depth_map: torch.Tensor):
        """
        Enhanced forward pass based on real underwater imaging formula:
        I = J•e^(-βDz) + B∞•(1 – e^(-βBz))
        
        depth_map: (H, W) or (1, 1, H, W), in meters (positive), cuda tensor
        returns:
            a_map: (3, H, W) - direct attenuation A(z) = e^(-βDz)
            B_map: (3, H, W) - backscatter B(z) = B∞•(1 – e^(-βBz))
        """
        if depth_map.dim() == 2:
            z = depth_map[None, None]  # (1,1,H,W)
        elif depth_map.dim() == 4:
            z = depth_map
        else:
            raise ValueError("depth_map must be (H,W) or (1,1,H,W)")

        if self.depth_conv is not None:
            z_feat = F.relu(self.depth_conv(z))
        else:
            z_feat = z

        # Ensure positive depth
        z_pos = torch.clamp(z_feat, min=0.0)
        
        # Reshape coefficient ranges for broadcasting
        beta_d = self.beta_d_range.view(1, self.num_basis, 1, 1)  # (1, K, 1, 1)
        beta_b = self.beta_b_range.view(1, self.num_basis, 1, 1)  # (1, K, 1, 1)

        # Compute exponential terms
        E_d = torch.exp(-beta_d * z_pos)        # Direct attenuation: e^(-βDz)
        E_b = torch.exp(-beta_b * z_pos)        # Backscatter attenuation: e^(-βBz)

        # Per-channel computation
        a_map_list = []
        B_map_list = []
        for c in range(3):
            # Direct attenuation: A(z) = e^(-βDz)
            wc = self.a_weights[c].view(1, self.num_basis, 1, 1)
            bc = self.a_bias[c]
            # Combine multiple basis functions for direct attenuation
            a_c = torch.exp((wc * E_d).sum(dim=1, keepdim=False) + bc)  # (1,H,W)
            a_c = torch.clamp(a_c, 0.0, 1.0)  # Ensure [0,1] range
            
            # Backscatter: B(z) = B∞•(1 – e^(-βBz))
            vc = self.b_weights[c].view(1, self.num_basis, 1, 1)
            uc = self.b_bias[c]
            b_inf_c = torch.sigmoid(self.b_infinity[c])  # B∞ for channel c
            
            # Compute backscatter component
            backscatter_term = (vc * (1.0 - E_b)).sum(dim=1, keepdim=False) + uc
            B_c = b_inf_c * torch.sigmoid(backscatter_term)  # (1,H,W)
            B_c = torch.clamp(B_c, 0.0, 1.0)  # Ensure [0,1] range
            
            a_map_list.append(a_c)
            B_map_list.append(B_c)

        a_map = torch.cat(a_map_list, dim=0)  # (3,H,W)
        B_map = torch.cat(B_map_list, dim=0)  # (3,H,W)
        return a_map, B_map

    @staticmethod
    def tv_loss(x: torch.Tensor):
        """ Total variation on (C,H,W). """
        dh = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean()
        dw = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
        return dh + dw

    # -------------------- IO helpers (match AppModel style) --------------------
    def save_weights(self, model_path: str, iteration: int):
        import os
        out_weights_path = os.path.join(model_path, "medium_model/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        print(f"save medium model. path: {out_weights_path}")
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'medium.pth'))

    def load_weights(self, model_path: str, iteration: int = -1):
        import os
        def _searchForMaxIteration(folder):
            try:
                saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
                return max(saved_iters)
            except Exception:
                return None
        if iteration == -1:
            folder = os.path.join(model_path, "medium_model")
            loaded_iter = _searchForMaxIteration(folder)
            if loaded_iter is None:
                raise FileNotFoundError("No saved medium_model weights found at {}".format(folder))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "medium_model/iteration_{}/medium.pth".format(loaded_iter))
        state_dict = torch.load(weights_path, map_location='cuda')
        self.load_state_dict(state_dict)


