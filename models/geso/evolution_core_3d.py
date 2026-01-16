# models/gseo/evolution_core_3d.py
"""
GSEO Evolution Core for 3D steady-state problems
"""

import torch
import torch.nn as nn
from .spectral_core import SpectralBlock
from .curvature_gate import DualCurvatureGate


class GSEOEvolution3D(nn.Module):
    """3D steady-state enhanced core"""
    
    def __init__(self, spatial_dim=3, embed_dim=256, spatial_num_blocks=6,
                 use_gating=True, boundary_type='dirichlet', 
                 gate_temperature=2.0, mode='steady', **kwargs):
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
        
        self.feature_amplifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        self.residual_scale = nn.Parameter(torch.ones(1) * 1.5)
        
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        if use_gating:
            self.gate = DualCurvatureGate(
                embed_dim=embed_dim,
                spatial_dim=spatial_dim,
                temperature=gate_temperature,
                boundary_type=boundary_type,
            )
        else:
            self.gate = None
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.feature_amplifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, current_features, physical_field=None,
                geometry_curvature=None, spatial_size=None, return_aux=False):
        aux_info = {}
        B, N, D = current_features.shape
        
        afno_feat = current_features
        for block in self.spatial_processor:
            afno_feat = block(afno_feat, spatial_size)
        afno_feat = self.spatial_norm(afno_feat)
        
        amplified_feat = self.feature_amplifier(afno_feat)
        
        delta = amplified_feat - current_features
        scale = self.residual_scale.clamp(min=0.5, max=3.0)
        delta = delta * scale
        
        delta = self.output_proj(delta)
        
        if self.use_gating and self.gate is not None:
            gate_weights, gate_info = self.gate(
                current_features, 
                physical_field=physical_field,
                geometry_curvature=geometry_curvature
            )
            delta = gate_weights * delta
            aux_info.update(gate_info)
        
        output = current_features + delta
        
        aux_info['delta_norm'] = delta.norm().item()
        aux_info['delta_ratio'] = (delta.norm() / current_features.norm()).item()
        aux_info['residual_scale'] = scale.item()
        
        if return_aux:
            return output, aux_info
        return output