"""
Grid Decoder: For structured grid outputs (1D/2D)
"""

import torch
import torch.nn as nn


class GridDecoder(nn.Module):
    """Grid decoder (pure linear projection)"""
    
    def __init__(self, embed_dim, out_channels, hidden_dim=None, spatial_dim=1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        
        self.decoder = nn.Linear(embed_dim, out_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, features, spatial_size=None):
        B, N, D = features.shape
        output = self.decoder(features)
        
        if spatial_size is not None and self.spatial_dim == 2:
            H, W = spatial_size
            output = output.reshape(B, H, W, self.out_channels)
        
        return output