"""
GSEO Core for 1D/2D transient problems
"""

import torch
import torch.nn as nn
from .spectral_core import SpectralBlock
from .curvature_gate import CurvatureGate


class GSEOCore(nn.Module):
    """Physics-informed AFNO core"""
    
    def __init__(self, spatial_dim, embed_dim, spatial_num_blocks=4,
                 use_gating=True, boundary_type='periodic', 
                 gate_temperature=2.0, mode='transient', **kwargs):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        self.use_gating = use_gating
        self.mode = mode
        
        self.spatial_processor = nn.ModuleList([
            SpectralBlock(dim=embed_dim, spatial_dim=spatial_dim, **kwargs)
            for _ in range(spatial_num_blocks)
        ])
        self.spatial_norm = nn.LayerNorm(embed_dim)
        
        if mode == 'transient':
            self.output_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            self.output_proj = nn.Identity()
        
        if use_gating:
            self.gate = CurvatureGate(
                embed_dim=embed_dim,
                spatial_dim=spatial_dim,
                temperature=gate_temperature,
                boundary_type=boundary_type,
            )
        else:
            self.gate = None
    
    def forward(self, current_features, physical_field=None,
                geometry_curvature=None, spatial_size=None, return_aux=False):
        aux_info = {}
        B, N, D = current_features.shape
        
        afno_feat = current_features
        for block in self.spatial_processor:
            afno_feat = block(afno_feat, spatial_size)
        afno_feat = self.spatial_norm(afno_feat)
        
        delta = self.output_proj(afno_feat - current_features)
        
        if self.use_gating and self.gate is not None:
            phys_field = self._prepare_physical_field(physical_field, spatial_size)
            gate_weights, gate_info = self.gate(current_features, phys_field)
            delta = gate_weights * delta
            aux_info.update(gate_info)
        
        aux_info['delta_norm'] = delta.norm().item()
        
        if self.mode == 'transient':
            output = delta
        else:
            output = current_features + delta
        
        if return_aux:
            return output, aux_info
        return output
    
    def _prepare_physical_field(self, physical_field, spatial_size):
        if physical_field is None:
            return None
        return physical_field