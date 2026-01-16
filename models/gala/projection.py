# models/gala/projection.py
"""
GALA Projection: Point cloud to regular grid projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn


class GALAProjection(nn.Module):
    """
    Point cloud to regular grid projection layer
    
    Method: k-nearest neighbor weighted projection
    """
    
    def __init__(self, point_dim, grid_dim, grid_size, k_neighbors=32):
        super().__init__()
        
        self.point_dim = point_dim
        self.grid_dim = grid_dim
        self.k = k_neighbors
        
        # Handle grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size,)
        self.grid_size = tuple(grid_size)
        self.spatial_dim = len(grid_size)
        
        # Create grid coordinates
        if self.spatial_dim == 2:
            self.D, self.H, self.W = 1, grid_size[0], grid_size[1]
            y = torch.linspace(0, 1, self.H)
            x = torch.linspace(0, 1, self.W)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            grid_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        elif self.spatial_dim == 3:
            self.D, self.H, self.W = grid_size
            z = torch.linspace(0, 1, self.D)
            y = torch.linspace(0, 1, self.H)
            x = torch.linspace(0, 1, self.W)
            zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
            grid_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        else:
            raise ValueError(f"Only 2D/3D supported, got spatial_dim={self.spatial_dim}")
        
        self.register_buffer('grid_coords', grid_coords)
        
        # Feature projection
        self.feature_proj = nn.Linear(point_dim, grid_dim)
        
        # Weight network (based on relative position + distance)
        self.weight_net = nn.Sequential(
            nn.Linear(self.spatial_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Output projection
        self.out_proj = nn.Linear(grid_dim, grid_dim)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(grid_dim)
    
    def forward(self, point_features, point_coords):
        """
        Forward pass
        
        Args:
            point_features: [N, point_dim] point cloud features
            point_coords: [N, spatial_dim] point cloud coordinates (normalized to [0,1])
        
        Returns:
            grid_features: [1, D, H, W, grid_dim] or [1, H, W, grid_dim]
        """
        N = point_features.shape[0]
        DHW = self.grid_coords.shape[0]
        
        # Project point features
        point_feat_proj = self.feature_proj(point_features)
        
        # k-NN assignment
        assign_index = knn(point_coords, self.grid_coords, self.k)
        grid_indices = assign_index[0]
        point_indices = assign_index[1]
        
        # Compute weights
        relative_pos = self.grid_coords[grid_indices] - point_coords[point_indices]
        distance = torch.norm(relative_pos, dim=-1, keepdim=True)
        weight_input = torch.cat([relative_pos, distance], dim=-1)
        weights = self.weight_net(weight_input).view(DHW, self.k)
        weights = F.softmax(weights, dim=-1).view(DHW * self.k, 1)
        
        # Weighted aggregation
        neighbor_feats = point_feat_proj[point_indices]
        weighted_feats = neighbor_feats * weights
        
        output = torch.zeros(DHW, self.grid_dim, device=point_features.device)
        output.scatter_add_(0, grid_indices.unsqueeze(-1).expand(-1, self.grid_dim), weighted_feats)
        
        # Output projection
        output = self.out_proj(output)
        output = self.output_norm(output)
        
        # Reshape to grid shape
        if self.spatial_dim == 2:
            return output.reshape(1, self.H, self.W, self.grid_dim)
        elif self.spatial_dim == 3:
            return output.reshape(1, self.D, self.H, self.W, self.grid_dim)