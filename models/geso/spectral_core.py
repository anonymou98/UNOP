# models/gseo/spectral_core.py
"""
GSEO Spectral Core: AFNO-based spectral filtering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralCore(nn.Module):
    """Spectral filtering core using AFNO"""
    
    def __init__(self, hidden_size, spatial_dim, num_blocks=8, 
                 hidden_size_factor=1, max_grid_size=1024, **kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.spatial_dim = spatial_dim
        self.num_blocks = num_blocks

        hidden_dim = hidden_size * hidden_size_factor
        
        self.fc1_real = nn.Linear(hidden_size, hidden_dim)
        self.fc1_imag = nn.Linear(hidden_size, hidden_dim)
        self.fc2_real = nn.Linear(hidden_dim, hidden_size)
        self.fc2_imag = nn.Linear(hidden_dim, hidden_size)
        
        self.max_freqs = max_grid_size // 2 + 1 
        self.max_grid = max_grid_size           

        scale = 0.02

        if spatial_dim == 1:
            self.emb_real_0 = nn.Parameter(torch.randn(1, self.max_freqs, hidden_size) * scale)
            self.emb_imag_0 = nn.Parameter(torch.randn(1, self.max_freqs, hidden_size) * scale)
        elif spatial_dim == 2:
            self.emb_real_0 = nn.Parameter(torch.randn(1, self.max_grid, 1, hidden_size) * scale)
            self.emb_imag_0 = nn.Parameter(torch.randn(1, self.max_grid, 1, hidden_size) * scale)
            self.emb_real_1 = nn.Parameter(torch.randn(1, 1, self.max_freqs, hidden_size) * scale)
            self.emb_imag_1 = nn.Parameter(torch.randn(1, 1, self.max_freqs, hidden_size) * scale)
        elif spatial_dim == 3:
            self.emb_real_0 = nn.Parameter(torch.randn(1, self.max_grid, 1, 1, hidden_size) * scale)
            self.emb_imag_0 = nn.Parameter(torch.randn(1, self.max_grid, 1, 1, hidden_size) * scale)
            self.emb_real_1 = nn.Parameter(torch.randn(1, 1, self.max_grid, 1, hidden_size) * scale)
            self.emb_imag_1 = nn.Parameter(torch.randn(1, 1, self.max_grid, 1, hidden_size) * scale)
            self.emb_real_2 = nn.Parameter(torch.randn(1, 1, 1, self.max_freqs, hidden_size) * scale)
            self.emb_imag_2 = nn.Parameter(torch.randn(1, 1, 1, self.max_freqs, hidden_size) * scale)

        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1_real, self.fc1_imag, self.fc2_real, self.fc2_imag]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, spatial_size):
        B = x.shape[0]
        C = x.shape[-1]
        
        if self.spatial_dim == 1:
            L = spatial_size[0]
            x = x.reshape(B, L, C)
            x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
            
            f_dim = x_fft.shape[1]
            x_real = x_fft.real + self.emb_real_0[:, :f_dim, :]
            x_imag = x_fft.imag + self.emb_imag_0[:, :f_dim, :]
            fft_size = L

        elif self.spatial_dim == 2:
            H, W = spatial_size
            x = x.reshape(B, H, W, C)
            x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
            
            H_curr, W_freq_curr = x_fft.shape[1], x_fft.shape[2]
            emb_r = self.emb_real_0[:, :H_curr, :, :] + self.emb_real_1[:, :, :W_freq_curr, :]
            emb_i = self.emb_imag_0[:, :H_curr, :, :] + self.emb_imag_1[:, :, :W_freq_curr, :]
            
            x_real = x_fft.real + emb_r
            x_imag = x_fft.imag + emb_i
            fft_size = (H, W)

        elif self.spatial_dim == 3:
            D, H, W = spatial_size
            x = x.reshape(B, D, H, W, C)
            x_fft = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
            
            D_c, H_c, W_fc = x_fft.shape[1], x_fft.shape[2], x_fft.shape[3]
            emb_r = (self.emb_real_0[:, :D_c, :, :, :] + 
                     self.emb_real_1[:, :, :H_c, :, :] + 
                     self.emb_real_2[:, :, :, :W_fc, :])
            emb_i = (self.emb_imag_0[:, :D_c, :, :, :] + 
                     self.emb_imag_1[:, :, :H_c, :, :] + 
                     self.emb_imag_2[:, :, :, :W_fc, :])
            
            x_real = x_fft.real + emb_r
            x_imag = x_fft.imag + emb_i
            fft_size = (D, H, W)
        else:
            raise ValueError(f"Unsupported spatial_dim: {self.spatial_dim}")
        
        orig_shape = x_real.shape
        x_real = x_real.reshape(-1, C)
        x_imag = x_imag.reshape(-1, C)
        
        h_real = F.gelu(self.fc1_real(x_real) - self.fc1_imag(x_imag))
        h_imag = F.gelu(self.fc1_real(x_imag) + self.fc1_imag(x_real))
        
        out_real = self.fc2_real(h_real) - self.fc2_imag(h_imag)
        out_imag = self.fc2_real(h_imag) + self.fc2_imag(h_real)
        
        out_real = out_real.reshape(orig_shape)
        out_imag = out_imag.reshape(orig_shape)
        
        x_fft = torch.complex(out_real, out_imag)
        
        if self.spatial_dim == 1:
            x = torch.fft.irfft(x_fft, n=fft_size, dim=1, norm='ortho')
            x = x.reshape(B, -1, C)
        elif self.spatial_dim == 2:
            x = torch.fft.irfft2(x_fft, s=fft_size, dim=(1, 2), norm='ortho')
            x = x.reshape(B, -1, C)
        elif self.spatial_dim == 3:
            x = torch.fft.irfftn(x_fft, s=fft_size, dim=(1, 2, 3), norm='ortho')
            x = x.reshape(B, -1, C)
        
        return x


class SpectralBlock(nn.Module):
    """Spectral block with LayerNorm and FFN"""
    
    def __init__(self, dim, spatial_dim, mlp_ratio=4., drop=0., 
                 num_blocks=8, max_grid_size=1024, **kwargs):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.filter = SpectralCore(
            hidden_size=dim,
            spatial_dim=spatial_dim,
            num_blocks=num_blocks,
            max_grid_size=max_grid_size, 
            **kwargs
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x, spatial_size):
        x = x + self.filter(self.norm1(x), spatial_size)
        x = x + self.mlp(self.norm2(x))
        return x