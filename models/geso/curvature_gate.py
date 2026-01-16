# models/gseo/curvature_gate.py
"""
GSEO Curvature Gate: Curvature-aware gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CurvatureGate(nn.Module):
    """Curvature-guided gate for 1D/2D transient problems"""
    
    def __init__(self, embed_dim, spatial_dim, temperature=1.0,
                 boundary_type='periodic', **kwargs):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.temperature = float(temperature)
        self.boundary_type = boundary_type
        
        self.curvature_scale = nn.Parameter(torch.tensor(1.0))
        self.curvature_bias = nn.Parameter(torch.tensor(-0.5))
        self.contrast_gamma = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, current_feat, physical_field, dt=None):
        B = current_feat.shape[0]
        
        if self.spatial_dim == 1:
            curvature = self._compute_curvature_1d(physical_field)
        elif self.spatial_dim == 2:
            N_target = current_feat.shape[1]
            curvature = self._compute_curvature_2d(physical_field, N_target)
        else:
            curvature = torch.ones(B, current_feat.shape[1], device=current_feat.device) * 0.5
        
        curvature_enhanced = self._enhance_curvature_contrast(curvature)
        
        effective_scale = self.curvature_scale.clamp(min=0.5, max=8.0)
        effective_bias = self.curvature_bias.clamp(min=-3.0, max=3.0)
        
        logit = effective_scale * curvature_enhanced + effective_bias
        gate_weights = torch.sigmoid(logit / self.temperature)
        
        info = {
            'curvature_mean': curvature.mean().item(),
            'gate_weight_mean': gate_weights.mean().item(),
        }
        
        return gate_weights.unsqueeze(-1), info
    
    def _enhance_curvature_contrast(self, curvature):
        gamma = self.contrast_gamma.clamp(min=0.3, max=1.5)
        curvature = torch.pow(curvature.clamp(min=1e-8), gamma)
        
        curv_min = curvature.min(dim=1, keepdim=True)[0]
        curv_max = curvature.max(dim=1, keepdim=True)[0]
        denom = (curv_max - curv_min).clamp(min=1e-6)
        curvature = (curvature - curv_min) / denom
        
        return curvature
    
    def _compute_curvature_1d(self, field):
        field = field.detach()
        
        if field.dim() == 3:
            u = field[..., 0]
        else:
            u = field
        
        B, N = u.shape
        
        if self.boundary_type == 'periodic':
            d2u = torch.roll(u, -1, dims=1) - 2 * u + torch.roll(u, 1, dims=1)
        else:
            d2u = torch.zeros_like(u)
            d2u[:, 1:-1] = u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]
            d2u[:, 0] = d2u[:, 1]
            d2u[:, -1] = d2u[:, -2]
        
        d2u_abs = d2u.abs()
        q95 = torch.quantile(d2u_abs, 0.95, dim=1, keepdim=True).clamp(min=1e-6)
        curvature = (d2u_abs / q95).clamp(max=1.0)
        
        return curvature
    
    def _compute_curvature_2d(self, field, N_target):
        field = field.detach()
        B = field.shape[0]
        
        if field.dim() == 3:
            N = field.shape[1]
            H = W = int(N ** 0.5)
            u = field[..., 0].reshape(B, H, W)
        elif field.dim() == 4:
            H, W = field.shape[1], field.shape[2]
            u = field[..., 0]
        else:
            raise ValueError(f"Unexpected field shape: {field.shape}")
        
        if self.boundary_type == 'periodic':
            d2u_dx2 = torch.roll(u, -1, dims=2) - 2 * u + torch.roll(u, 1, dims=2)
            d2u_dy2 = torch.roll(u, -1, dims=1) - 2 * u + torch.roll(u, 1, dims=1)
        else:
            d2u_dx2 = torch.zeros_like(u)
            d2u_dy2 = torch.zeros_like(u)
            d2u_dx2[:, :, 1:-1] = u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]
            d2u_dy2[:, 1:-1, :] = u[:, 2:, :] - 2 * u[:, 1:-1, :] + u[:, :-2, :]
        
        laplacian = (d2u_dx2.abs() + d2u_dy2.abs()).reshape(B, -1)
        
        q95 = torch.quantile(laplacian, 0.95, dim=1, keepdim=True).clamp(min=1e-6)
        curvature = (laplacian / q95).clamp(max=1.0)
        
        return curvature


class DualCurvatureGate(nn.Module):
    """Dual-curvature gate for 3D steady-state problems"""
    
    def __init__(self, embed_dim, spatial_dim=3, temperature=1.0,
                 init_geom_weight=0.7, init_phys_weight=0.3,
                 boundary_type='dirichlet', **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        self.temperature = temperature
        self.boundary_type = boundary_type
        
        self.w_geom = nn.Parameter(torch.tensor(init_geom_weight))
        self.w_phys = nn.Parameter(torch.tensor(init_phys_weight))
        
        self.gate_scale = nn.Parameter(torch.tensor(2.0))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, current_feat, physical_field=None, geometry_curvature=None, dt=None):
        B, N, D = current_feat.shape
        device = current_feat.device
        
        if geometry_curvature is not None:
            geom_curv = self._process_geometry_curvature(geometry_curvature, B, N, device)
            has_geom = True
        else:
            geom_curv = torch.zeros(B, N, device=device)
            has_geom = False
        
        if physical_field is not None:
            phys_curv = self._compute_physics_curvature(physical_field, N)
        else:
            phys_curv = self._compute_feature_curvature(current_feat)
        
        phys_curv = self._normalize(phys_curv)
        
        if has_geom:
            w_g = torch.sigmoid(self.w_geom)
            w_p = torch.sigmoid(self.w_phys)
            total_w = w_g + w_p + 1e-8
            fused_curvature = (w_g * geom_curv + w_p * phys_curv) / total_w
        else:
            fused_curvature = phys_curv
        
        scale = self.gate_scale.clamp(min=0.5, max=5.0)
        bias = self.gate_bias.clamp(min=-3.0, max=3.0)
        
        logit = scale * fused_curvature + bias
        gate_weights = torch.sigmoid(logit / self.temperature)
        
        info = {
            'geom_curv_mean': geom_curv.mean().item() if has_geom else 0.0,
            'phys_curv_mean': phys_curv.mean().item(),
            'gate_mean': gate_weights.mean().item(),
        }
        
        return gate_weights.unsqueeze(-1), info
    
    def _process_geometry_curvature(self, curvature, B, N, device):
        curv = curvature.squeeze(-1)
        if curv.dim() == 1:
            curv = curv.unsqueeze(0).expand(B, -1)
        
        if curv.shape[1] != N:
            curv = F.interpolate(
                curv.unsqueeze(1), size=N, mode='linear', align_corners=True
            ).squeeze(1)
        
        return self._normalize(curv)
    
    def _compute_physics_curvature(self, grid_physics, N_target):
        B = grid_physics.shape[0]
        device = grid_physics.device
        
        if grid_physics.dim() != 5:
            return torch.ones(B, N_target, device=device) * 0.5
        
        u = grid_physics[..., 0]
        
        d2u_dx = u[:, :, :, 2:] - 2*u[:, :, :, 1:-1] + u[:, :, :, :-2]
        d2u_dy = u[:, :, 2:, :] - 2*u[:, :, 1:-1, :] + u[:, :, :-2, :]
        d2u_dz = u[:, 2:, :, :] - 2*u[:, 1:-1, :, :] + u[:, :-2, :, :]
        
        d2u_dx = F.pad(d2u_dx, (1, 1, 0, 0, 0, 0), mode='replicate')
        d2u_dy = F.pad(d2u_dy, (0, 0, 1, 1, 0, 0), mode='replicate')
        d2u_dz = F.pad(d2u_dz, (0, 0, 0, 0, 1, 1), mode='replicate')
        
        laplacian = (d2u_dx.abs() + d2u_dy.abs() + d2u_dz.abs())
        curv_flat = laplacian.reshape(B, -1)
        
        if curv_flat.shape[1] != N_target:
            curv_flat = F.interpolate(
                curv_flat.unsqueeze(1), size=N_target, mode='linear', align_corners=True
            ).squeeze(1)
        
        return curv_flat
    
    def _compute_feature_curvature(self, features):
        return features.var(dim=-1)
    
    def _normalize(self, x):
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8)