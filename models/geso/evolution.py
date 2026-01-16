"""
GSEO Evolution Operator for transient problems (1D/2D)
"""

import torch
import torch.nn as nn
from .evolution_core import GSEOCore


class GSEOEvolution(nn.Module):
    """
    Evolution Operator: Euler integration z_{t+dt} = z_t + dt * F(z_t)
    """
    
    def __init__(self, embed_dim, spatial_dim, **kwargs):
        super().__init__()
        
        self.core = GSEOCore(
            spatial_dim=spatial_dim,
            embed_dim=embed_dim,
            **kwargs
        )
        
        self.log_dt_scale = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, z, dt=None, physical_field=None, spatial_size=None):
        B = z.shape[0]
        device = z.device
        
        if dt is None:
            dt = torch.ones(B, 1, device=device)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        
        velocity, aux_info = self.core(
            current_features=z,
            physical_field=physical_field,
            spatial_size=spatial_size,
            return_aux=True
        )
        
        dt_scale = torch.exp(self.log_dt_scale).clamp(min=0.1, max=10.0)
        effective_dt = dt * dt_scale
        
        delta_z = effective_dt.unsqueeze(-1) * velocity
        
        aux_info['dt_scale'] = dt_scale.item()
        aux_info['effective_dt_mean'] = effective_dt.mean().item()
        aux_info['delta_z_norm'] = delta_z.norm().item()
        
        return delta_z, aux_info