import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
      
      
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 3, 3), device=means3D.device)
        ### FILL:
        ### J_proj = ...


        J_proj[:, 0, 0] = K[0,0]/cam_points[:,2]  # dx/dX
        J_proj[:, 0, 1] = 0  # dx/dY
        J_proj[:, 0, 2] = -(K[0,0]*cam_points[:,0])/(cam_points[:,2]*cam_points[:,2])  # dx/dZ
        J_proj[:, 1, 0] = 0  # dy/dX
        J_proj[:, 1, 1] = K[1,1] / cam_points[:,2]  # dy/dY
        J_proj[:, 1, 2] = -(K[1,1]*cam_points[:,1]) / (cam_points[:,2]*cam_points[:,2])   # dy/dZ
        
        
        
        
        # Transform covariance to camera space
        ### FILL: Aplly world to camera rotation to the 3d covariance matrix
        ### covs_cam = ...  # (N, 3, 3)
        covs_cam = torch.matmul(R, torch.matmul(covs3d, R.T))
        
        # Project to 2D
        covs2D = torch.bmm(J_proj.permute(0, 2, 1), torch.bmm(covs_cam, J_proj))  # (N, 2, 2)
        
        
        
        return means2D, covs2D[:,:2,:2], depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute determinant for normalization
        ### FILL: compute the gaussian values
        ### gaussian = ... ## (N, H, W)
        
        det_covs = torch.det(covs2D)  # (N,)
        inv_covs = torch.linalg.inv(covs2D)  # (N, 2, 2)

        # Compute the Mahalanobis distance (N, H, W)
        dx_cov_inv = torch.einsum('nhwi,nij->nhwj', dx, inv_covs)  # (N, H, W, 2)

        mahalanobis = torch.sum(dx_cov_inv * dx, dim=-1)  # (N, H, W)
        
        # Compute Gaussian values (N, H, W)
        gaussian = (1. / (2 * torch.pi * det_covs.sqrt())).unsqueeze(-1).unsqueeze(-1) * torch.exp(-0.5 * mahalanobis)  # (N, H, W)

        # gaussian = torch.exp(-0.5 * mahalanobis)  # (N, H, W)

        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights
        ### FILL:
        ### weights = ... # (N, H, W)
        # weights=alphas*torch.cumprod(1-alphas,dim=0)
        # weight_cum = torch.cumprod(1 - alphas , dim=0)  # (N, H, W)
        # weights = alphas * weight_cum  # (N, H, W)
        T = torch.cumprod(torch.cat([torch.ones(1, self.H, self.W, device=alphas.device), 1 - alphas[:-1]], dim=0), dim=0)
        weights = alphas * T
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
