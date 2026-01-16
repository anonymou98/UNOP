# models/decoder/point_sampler.py
"""
Point Cloud Sampler: Grid to point cloud decoding (3D)

原文件: models/decoders/sampler.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudSampler(nn.Module):
    """
    网格到点云的可微采样解码器（跳跃连接版）
    
    原名: GridToPointCloudSampler
    """
    
    def __init__(self, grid_dim, out_dim, hidden_dim=256, spatial_dim=3, 
                 num_layers=3, use_residual=True, dropout=0.0,
                 use_skip_connection=True):
        super().__init__()
        
        self.grid_dim = grid_dim
        self.out_dim = out_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.use_skip_connection = use_skip_connection
        
        # ============== 以下完全保留你的原始代码 ==============
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(grid_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.velocity_decoder = self._build_decoder_head(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=num_layers,
        )
        
        self.pressure_decoder = self._build_decoder_head(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
        )
        
        if use_skip_connection:
            self.skip_velocity = nn.Sequential(
                nn.Linear(grid_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 3)
            )
            self.skip_pressure = nn.Sequential(
                nn.Linear(grid_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.skip_weight = nn.Parameter(torch.tensor(0.3))
        
        self.vel_scale = nn.Parameter(torch.ones(3) * 3.0)
        self.pres_scale = nn.Parameter(torch.ones(1) * 3.5)
        self.vel_bias = nn.Parameter(torch.zeros(3))
        self.pres_bias = nn.Parameter(torch.zeros(1))
        
        if use_residual and grid_dim != hidden_dim:
            self.residual_proj = nn.Linear(grid_dim, hidden_dim)
        else:
            self.residual_proj = None
        
        self._init_weights()
    
    def _build_decoder_head(self, input_dim, hidden_dim, output_dim, num_layers):
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.GELU(),
                nn.LayerNorm(next_dim),
            ])
            current_dim = next_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for module in [self.shared_encoder, self.velocity_decoder, self.pressure_decoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        for decoder in [self.velocity_decoder, self.pressure_decoder]:
            last_layer = decoder[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.normal_(last_layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(last_layer.bias)
        
        if self.use_skip_connection:
            for module in [self.skip_velocity, self.skip_pressure]:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.5)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        
        if self.residual_proj is not None:
            nn.init.orthogonal_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
    
    def forward(self, grid_features, query_coords):
        
        B, D, H, W, C = grid_features.shape
        grid_cf = grid_features.permute(0, 4, 1, 2, 3).contiguous()
        
        if query_coords.dim() == 2:
            N = query_coords.shape[0]
            coords = query_coords.unsqueeze(0).expand(B, -1, -1)
            squeeze_output = True
        else:
            coords = query_coords
            squeeze_output = False
        
        sampled = self._grid_sample_3d(grid_cf, coords)
        sampled_centered = sampled - sampled.mean(dim=1, keepdim=True)
        
        shared_feat = self.shared_encoder(sampled_centered)
        
        if self.use_residual and self.residual_proj is not None:
            residual = self.residual_proj(sampled_centered)
            shared_feat = shared_feat + 0.1 * residual
        
        velocity_main = self.velocity_decoder(shared_feat)
        pressure_main = self.pressure_decoder(shared_feat)
        
        if self.use_skip_connection:
            velocity_skip = self.skip_velocity(sampled_centered)
            pressure_skip = self.skip_pressure(sampled_centered)
            
            skip_w = torch.sigmoid(self.skip_weight)
            velocity_raw = (1 - skip_w) * velocity_main + skip_w * velocity_skip
            pressure_raw = (1 - skip_w) * pressure_main + skip_w * pressure_skip
        else:
            velocity_raw = velocity_main
            pressure_raw = pressure_main
        
        velocity = velocity_raw * self.vel_scale + self.vel_bias
        pressure = pressure_raw * self.pres_scale + self.pres_bias
        
        output = torch.cat([velocity, pressure], dim=-1)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
    
    def _grid_sample_3d(self, grid_cf, coords):
        coords_5d = coords.unsqueeze(2).unsqueeze(2)
        coords_normalized = (coords_5d * 2 - 1).flip(-1)
        
        sampled = F.grid_sample(
            grid_cf, coords_normalized,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return sampled
    
    def get_scale_info(self):
        info = {
            'vel_scale': self.vel_scale.detach().cpu().numpy(),
            'pres_scale': self.pres_scale.item(),
            'vel_bias': self.vel_bias.detach().cpu().numpy(),
            'pres_bias': self.pres_bias.item(),
        }
        if self.use_skip_connection:
            info['skip_weight'] = torch.sigmoid(self.skip_weight).item()
        return info


