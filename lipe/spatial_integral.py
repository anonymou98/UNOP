# lipe/spatial_integral.py
"""
LIPE Spatial Integral Loss (3D Steady-State)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LIPESpatialLoss(nn.Module):
    """
    3D Spatial Integral Loss for Steady-State Problems
    """
    
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        
        # Weights
        self.w_data = config.get('w_data', 1.0)
        self.w_mass_soft = config.get('w_mass_soft', 0.1)
        self.w_momentum = config.get('w_momentum', 0.1)
        self.w_smoothness = config.get('w_smoothness', 0.01)
        self.w_bc = config.get('w_bc', 0.5)
        
        # Sampling parameters
        self.num_control_volumes = config.get('num_control_volumes', 64)
        self.num_surface_points = config.get('num_surface_points', 128)
        self.radius_range = config.get('radius_range', (0.08, 0.20))
        self.mass_tolerance = config.get('mass_tolerance', 0.01)
        
    def forward(self, pred, target=None, grid_physics=None, 
                surf_mask=None, normal=None, **kwargs):
        """
        Compute LIPE spatial loss
        """
        device = pred.device
        loss_dict = {'data': 0.0, 'mass': 0.0, 'momentum': 0.0, 'bc': 0.0}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        pred_2d = pred.reshape(-1, pred.shape[-1]) if pred.dim() == 3 else pred
        target_2d = target.reshape(-1, target.shape[-1]) if target is not None and target.dim() == 3 else target
        
        # 1. Data loss
        if self.w_data > 0 and target is not None:
            loss_data = F.mse_loss(pred_2d, target_2d)
            total_loss = total_loss + self.w_data * loss_data
            loss_dict['data'] = loss_data.item()
        
        # 2. Mass conservation
        if self.w_mass_soft > 0 and grid_physics is not None:
            loss_mass = self._soft_mass_conservation(grid_physics)
            total_loss = total_loss + self.w_mass_soft * loss_mass
            loss_dict['mass'] = loss_mass.item()
        
        # 3. Momentum conservation
        if self.w_momentum > 0 and grid_physics is not None:
            loss_mom = self._momentum_conservation(grid_physics)
            total_loss = total_loss + self.w_momentum * loss_mom
            loss_dict['momentum'] = loss_mom.item()
        
        # 4. Boundary condition
        if self.w_bc > 0 and surf_mask is not None and normal is not None:
            surf_mask_1d = surf_mask.reshape(-1) if surf_mask.dim() > 1 else surf_mask
            normal_2d = normal.reshape(-1, 3) if normal.dim() > 2 else normal
            
            loss_bc = self._boundary_condition(pred_2d, surf_mask_1d, normal_2d)
            total_loss = total_loss + self.w_bc * loss_bc
            loss_dict['bc'] = loss_bc.item()
        
        # 5. Smoothness
        if self.w_smoothness > 0 and grid_physics is not None:
            loss_smooth = self._smoothness_loss(grid_physics)
            total_loss = total_loss + self.w_smoothness * loss_smooth
            loss_dict['smoothness'] = loss_smooth.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _boundary_condition(self, pred, surf_mask, normal):
        """No-penetration BC: u·n = 0"""
        if surf_mask.dtype != torch.bool:
            surf_mask = surf_mask > 0.5
        
        n_surf = surf_mask.sum().item()
        if n_surf == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        N = pred.shape[0]
        if surf_mask.shape[0] != N or normal.shape[0] != N:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        velocity = pred[:, :3]
        surf_velocity = velocity[surf_mask]
        surf_normal = normal[surf_mask]
        surf_normal = F.normalize(surf_normal, dim=-1)
        
        u_normal = (surf_velocity * surf_normal).sum(dim=-1)
        loss_bc = (u_normal ** 2).mean()
        
        return loss_bc
    
    def _soft_mass_conservation(self, grid_physics):
        """Soft mass conservation via control volume sampling"""
        B, D, H, W, C = grid_physics.shape
        device = grid_physics.device
        velocity = grid_physics[..., :3]
        
        vel_scale = velocity.abs().mean().clamp(min=1e-6)
        
        violations = []
        for _ in range(self.num_control_volumes):
            radius = torch.empty(1, device=device).uniform_(*self.radius_range).item()
            net_flux = self._compute_sphere_flux(velocity, radius, B, device)
            
            normalized_violation = net_flux.abs() / vel_scale
            soft_part = F.relu(normalized_violation - self.mass_tolerance)
            hard_part = normalized_violation * 0.1
            
            violations.append((soft_part + hard_part).mean())
        
        return torch.stack(violations).mean()
    
    def _compute_sphere_flux(self, velocity, radius, B, device):
        """Compute net flux through spherical surface"""
        num_pts = self.num_surface_points
        
        margin = radius + 0.02
        center = torch.rand(B, 3, device=device) * (1 - 2*margin) + margin
        
        phi = torch.rand(B, num_pts, device=device) * 2 * math.pi
        cos_theta = torch.rand(B, num_pts, device=device) * 2 - 1
        sin_theta = torch.sqrt((1 - cos_theta**2).clamp(min=1e-8))
        
        normal = torch.stack([
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            cos_theta
        ], dim=-1)
        
        sample_coords = center.unsqueeze(1) + normal * radius
        grid_coords = sample_coords * 2 - 1
        grid_coords = grid_coords.flip(-1).view(B, num_pts, 1, 1, 3)
        
        vel_5d = velocity.permute(0, 4, 1, 2, 3)
        sampled_vel = F.grid_sample(
            vel_5d, grid_coords,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        sampled_vel = sampled_vel.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        flux = (sampled_vel * normal).sum(dim=-1)
        surface_area = 4 * math.pi * radius**2
        net_flux = flux.mean(dim=-1) * surface_area
        
        return net_flux
    
    def _momentum_conservation(self, grid_physics):
        """Momentum conservation"""
        B, D, H, W, C = grid_physics.shape
        device = grid_physics.device
        
        velocity = grid_physics[..., :3]
        pressure = grid_physics[..., 3:4]
        
        vel_scale = velocity.abs().mean().clamp(min=1e-6)
        pres_scale = pressure.abs().mean().clamp(min=1e-6)
        
        violations = []
        num_samples = max(16, self.num_control_volumes // 2)
        
        for _ in range(num_samples):
            radius = torch.empty(1, device=device).uniform_(*self.radius_range).item()
            
            margin = radius + 0.02
            center = torch.rand(B, 3, device=device) * (1 - 2*margin) + margin
            
            num_pts = self.num_surface_points
            phi = torch.rand(B, num_pts, device=device) * 2 * math.pi
            cos_theta = torch.rand(B, num_pts, device=device) * 2 - 1
            sin_theta = torch.sqrt((1 - cos_theta**2).clamp(min=1e-8))
            
            normal = torch.stack([
                sin_theta * torch.cos(phi),
                sin_theta * torch.sin(phi),
                cos_theta
            ], dim=-1)
            
            sample_coords = center.unsqueeze(1) + normal * radius
            grid_coords = (sample_coords * 2 - 1).flip(-1).view(B, num_pts, 1, 1, 3)
            
            vel_5d = velocity.permute(0, 4, 1, 2, 3)
            pres_5d = pressure.permute(0, 4, 1, 2, 3)
            
            sampled_vel = F.grid_sample(vel_5d, grid_coords, mode='bilinear',
                                        padding_mode='border', align_corners=True)
            sampled_pres = F.grid_sample(pres_5d, grid_coords, mode='bilinear',
                                         padding_mode='border', align_corners=True)
            
            sampled_vel = sampled_vel.squeeze(-1).squeeze(-1).permute(0, 2, 1)
            sampled_pres = sampled_pres.squeeze(-1).squeeze(-1).permute(0, 2, 1)
            
            v_dot_n = (sampled_vel * normal).sum(dim=-1, keepdim=True)
            momentum_flux = sampled_vel * v_dot_n + sampled_pres * normal
            
            surface_area = 4 * math.pi * radius**2
            net_momentum = momentum_flux.mean(dim=1) * surface_area
            
            violation = (net_momentum / (vel_scale * vel_scale + pres_scale)).norm(dim=-1)
            violations.append(violation.mean())
        
        return torch.stack(violations).mean()
    
    def _smoothness_loss(self, grid_physics):
        """Smoothness constraint"""
        velocity = grid_physics[..., :3]
        pressure = grid_physics[..., 3:4]
        
        dv_dx = velocity[:, :, :, 1:, :] - velocity[:, :, :, :-1, :]
        dv_dy = velocity[:, :, 1:, :, :] - velocity[:, :, :-1, :, :]
        dv_dz = velocity[:, 1:, :, :, :] - velocity[:, :-1, :, :, :]
        
        dp_dx = pressure[:, :, :, 1:, :] - pressure[:, :, :, :-1, :]
        dp_dy = pressure[:, :, 1:, :, :] - pressure[:, :, :-1, :, :]
        dp_dz = pressure[:, 1:, :, :, :] - pressure[:, :-1, :, :, :]
        
        smooth_vel = (dv_dx.pow(2).mean() + dv_dy.pow(2).mean() + dv_dz.pow(2).mean()) / 3
        smooth_pres = (dp_dx.pow(2).mean() + dp_dy.pow(2).mean() + dp_dz.pow(2).mean()) / 3
        
        return smooth_vel + 0.1 * smooth_pres