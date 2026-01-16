# models/gala/grid_encoder.py
"""
GALA Grid Encoder: For structured grid inputs (1D/2D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GALAGridEncoder(nn.Module):
    """
    GALA encoder for structured grids
    
    Encodes grid-based physical fields into latent representations
    with hybrid periodic position encoding.
    """
    
    def __init__(
        self,
        spatial_dim,
        in_channels,
        embed_dim,
        use_positional_encoding=True,
        pos_encoding_type='hybrid_periodic',
        padding_mode='zeros',
        grid_size=None
    ):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding_type = pos_encoding_type
        self.padding_mode = padding_mode
        self._grid_size = None
        
        # ========== Convolutional Encoder ==========
        if spatial_dim == 1:
            self.conv1_block = nn.Sequential(
                nn.Conv1d(in_channels, embed_dim // 2, kernel_size=5, padding=0),
                nn.GELU(),
                nn.GroupNorm(8, embed_dim // 2)
            )
            self.conv2 = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=0)
            
        elif spatial_dim == 2:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, kernel_size=5, padding=0),
                nn.GELU(),
                nn.GroupNorm(8, embed_dim // 2)
            )
            self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=0)
        
        # ========== Position Encoding ==========
        self.pos_embed_residual = None
        self.pos_embed_scale = None
        
        if use_positional_encoding and grid_size is not None:
            if isinstance(grid_size, int):
                N = grid_size
            elif isinstance(grid_size, (list, tuple)):
                N = grid_size[0] if len(grid_size) == 1 else grid_size[0] * grid_size[1]
            else:
                N = None

            if N is not None:
                self._grid_size = N
                
                if pos_encoding_type == 'hybrid_periodic':
                    self.pos_embed_residual = nn.Parameter(torch.zeros(1, N, embed_dim))
                    nn.init.trunc_normal_(self.pos_embed_residual, std=0.01)
                    self.pos_embed_scale = nn.Parameter(torch.tensor(0.1))
                    
                elif pos_encoding_type == 'learnable':
                    self.pos_embed_residual = nn.Parameter(torch.zeros(1, N, embed_dim))
                    nn.init.trunc_normal_(self.pos_embed_residual, std=0.02)
    
    def _pad(self, x, pad_size):
        """Universal padding"""
        mode_map = {
            'zeros': 'constant',
            'constant': 'constant',
            'replicate': 'replicate',
            'circular': 'circular',
            'reflect': 'reflect'
        }
        torch_mode = mode_map.get(self.padding_mode, 'constant')
        
        if self.spatial_dim == 1:
            pad_arg = (pad_size, pad_size)
        elif self.spatial_dim == 2:
            pad_arg = (pad_size, pad_size, pad_size, pad_size)
        else:
            return x
        
        if torch_mode == 'constant':
            return F.pad(x, pad_arg, mode='constant', value=0)
        else:
            return F.pad(x, pad_arg, mode=torch_mode)

    def forward(self, x, spatial_size=None):
        if self.spatial_dim == 1:
            if x.dim() == 3 and x.shape[1] != self.in_channels:
                x = x.permute(0, 2, 1).contiguous()
            B, C, N = x.shape
            spatial_size = (N,) if spatial_size is None else spatial_size
        elif self.spatial_dim == 2:
            if x.dim() == 4 and x.shape[1] != self.in_channels:
                x = x.permute(0, 3, 1, 2).contiguous()
            B, C, H, W = x.shape
            N = H * W
            spatial_size = (H, W) if spatial_size is None else spatial_size
        
        x = self._pad(x, 2)
        x = self.conv1_block(x)
        x = self._pad(x, 1)
        x = self.conv2(x)
        
        if self.spatial_dim == 1:
            x = x.transpose(1, 2)
        elif self.spatial_dim == 2:
            x = x.permute(0, 2, 3, 1).reshape(B, N, self.embed_dim).contiguous()
        
        if self.use_positional_encoding:
            pos_embed = self._get_positional_encoding(x.shape, spatial_size)
            x = x + pos_embed
        
        return x
    
    def _get_positional_encoding(self, shape, spatial_size):
        B, N, D = shape
        device = next(self.parameters()).device
        
        if self.pos_encoding_type == 'hybrid_periodic':
            fourier_base = self._get_fourier_periodic_base(N, D, device)
            
            if self.pos_embed_residual is not None:
                residual = self.pos_embed_residual
                
                if residual.shape[1] != N:
                    residual = F.interpolate(
                        residual.transpose(1, 2),
                        size=N,
                        mode='linear',
                        align_corners=True
                    ).transpose(1, 2)
                
                residual = self._make_periodic_residual(residual)
                scale = torch.sigmoid(self.pos_embed_scale)
                pos_embed = fourier_base + scale * residual
            else:
                pos_embed = fourier_base
            
            return pos_embed
        
        elif self.pos_encoding_type == 'learnable':
            if self.pos_embed_residual is not None:
                pos = self.pos_embed_residual
                
                if pos.shape[1] != N:
                    pos = F.interpolate(
                        pos.transpose(1, 2),
                        size=N,
                        mode='linear',
                        align_corners=True
                    ).transpose(1, 2)
                
                if self.padding_mode == 'circular':
                    pos = self._make_periodic_residual(pos)
                
                return pos
            else:
                return torch.zeros(1, N, D, device=device)
        
        elif self.pos_encoding_type == 'sinusoidal':
            if self.spatial_dim == 1:
                if self.padding_mode == 'circular':
                    return self._get_fourier_periodic_base(N, D, device)
                else:
                    return self._get_sinusoidal_encoding_1d(N, D, device)
            elif self.spatial_dim == 2:
                H, W = spatial_size
                return self._get_sinusoidal_encoding_2d(H, W, D, device)
        
        return torch.zeros(1, N, D, device=device)
    
    def _get_fourier_periodic_base(self, N, D, device):
        """Periodic Fourier basis with integer frequencies"""
        theta = torch.linspace(0, 2 * math.pi * (1 - 1/N), N, device=device)
        theta = theta.unsqueeze(1)
        
        num_freqs = D // 2
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float, device=device)
        freqs = freqs.unsqueeze(0)
        
        angles = theta * freqs
        
        pos_embed = torch.zeros(1, N, D, device=device)
        pos_embed[0, :, 0::2] = torch.sin(angles)
        pos_embed[0, :, 1::2] = torch.cos(angles)
        
        return pos_embed
    
    def _make_periodic_residual(self, residual):
        """Make learnable residual periodic (smooth boundary transition)"""
        N = residual.shape[1]
        
        if N < 4:
            return residual
        
        n_blend = max(1, N // 10)
        result = residual.clone()
        
        for i in range(n_blend):
            t = (i + 1) / (n_blend + 1)
            right_idx = N - n_blend + i
            result[:, right_idx, :] = (1 - t) * residual[:, right_idx, :] + t * residual[:, 0, :]
        
        return result
    
    def _get_sinusoidal_encoding_1d(self, N, D, device):
        """Standard 1D sinusoidal position encoding"""
        position = torch.arange(N, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=device) * 
                            -(math.log(10000.0) / D))
        pos_embed = torch.zeros(1, N, D, device=device)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embed
    
    def _get_sinusoidal_encoding_2d(self, H, W, D, device):
        """2D sinusoidal position encoding"""
        assert D % 4 == 0
        D_half = D // 2
        
        y_pos = torch.arange(H, dtype=torch.float, device=device).unsqueeze(1)
        y_div = torch.exp(torch.arange(0, D_half, 2, dtype=torch.float, device=device) * 
                         -(math.log(10000.0) / D_half))
        y_embed = torch.zeros(H, D_half, device=device)
        y_embed[:, 0::2] = torch.sin(y_pos * y_div)
        y_embed[:, 1::2] = torch.cos(y_pos * y_div)
        
        x_pos = torch.arange(W, dtype=torch.float, device=device).unsqueeze(1)
        x_div = torch.exp(torch.arange(0, D_half, 2, dtype=torch.float, device=device) * 
                         -(math.log(10000.0) / D_half))
        x_embed = torch.zeros(W, D_half, device=device)
        x_embed[:, 0::2] = torch.sin(x_pos * x_div)
        x_embed[:, 1::2] = torch.cos(x_pos * x_div)
        
        pos_embed = torch.zeros(1, H, W, D, device=device)
        pos_embed[0, :, :, :D_half] = y_embed.unsqueeze(1).repeat(1, W, 1)
        pos_embed[0, :, :, D_half:] = x_embed.unsqueeze(0).repeat(H, 1, 1)
        
        return pos_embed.reshape(1, H * W, D)