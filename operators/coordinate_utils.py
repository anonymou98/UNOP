# ============================================================================
# operators/coordinate_utils.py
# 坐标变换和归一化工具
# ============================================================================
"""
功能：
- 坐标归一化器
- 坐标变换（笛卡尔 ↔ 极坐标等）
"""

import torch
import numpy as np


class CoordinateNormalizer:
    """
    坐标归一化器
    """
    def __init__(self):
        self.min_coords = None
        self.max_coords = None
    
    def fit(self, coords):
        """
        拟合归一化参数
        
        Args:
            coords: [N, D] 坐标张量（D是维度）
        """
        if coords.dim() == 3:  # [B, N, D]
            coords = coords.reshape(-1, coords.shape[-1])
        
        self.min_coords = coords.min(dim=0)[0].detach().cpu()
        self.max_coords = coords.max(dim=0)[0].detach().cpu()
    
    def transform(self, coords):
        if self.min_coords is None:
            self.fit(coords.reshape(-1, coords.shape[-1]))
        
        min_coords = self.min_coords.to(coords.device)
        max_coords = self.max_coords.to(coords.device)
        
        normalized = (coords - min_coords) / (max_coords - min_coords + 1e-8)
        return torch.clamp(normalized, 0.0, 1.0)

    def normalize(self, coords):
        """transform 的别名"""
        return self.transform(coords)
    def inverse_transform(self, coords_normalized):
        """
        反归一化
        
        Args:
            coords_normalized: [..., D] 归一化后的坐标
        """
        if self.min_coords is None:
            raise RuntimeError("必须先调用 fit() 拟合归一化参数")
        
        min_coords = self.min_coords.to(coords_normalized.device)
        max_coords = self.max_coords.to(coords_normalized.device)
        
        return coords_normalized * (max_coords - min_coords) + min_coords

    def denormalize(self, coords_normalized):
        """inverse_transform 的别名"""
        return self.inverse_transform(coords_normalized)
def cartesian_to_polar(coords):
    """
    笛卡尔坐标 → 极坐标（2D）
    
    Args:
        coords: [..., 2] (x, y)
    
    Returns:
        polar: [..., 2] (r, θ)
    """
    x, y = coords[..., 0], coords[..., 1]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return torch.stack([r, theta], dim=-1)


def polar_to_cartesian(polar):
    """
    极坐标 → 笛卡尔坐标（2D）
    
    Args:
        polar: [..., 2] (r, θ)
    
    Returns:
        coords: [..., 2] (x, y)
    """
    r, theta = polar[..., 0], polar[..., 1]
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def cartesian_to_spherical(coords):
    """
    笛卡尔坐标 → 球坐标（3D）
    
    Args:
        coords: [..., 3] (x, y, z)
    
    Returns:
        spherical: [..., 3] (r, θ, φ)
    """
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(torch.sqrt(x**2 + y**2), z)  # 极角
    phi = torch.atan2(y, x)  # 方位角
    return torch.stack([r, theta, phi], dim=-1)


def spherical_to_cartesian(spherical):
    """
    球坐标 → 笛卡尔坐标（3D）
    
    Args:
        spherical: [..., 3] (r, θ, φ)
    
    Returns:
        coords: [..., 3] (x, y, z)
    """
    r, theta, phi = spherical[..., 0], spherical[..., 1], spherical[..., 2]
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)