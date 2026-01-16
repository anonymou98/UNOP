# lipe/__init__.py
"""
LIPE: Latent Integral Physics Embedding

Unsupervised learning via stochastic integration, replacing
differential residuals with integral constraints.
"""

from .temporal_integral_1d import LIPETemporalLoss1D
from .temporal_integral_2d import LIPETemporalLoss2D
from .spatial_integral import LIPESpatialLoss


def build_lipe_loss(problem_type, config):
    """
    Factory function for LIPE loss
    """
    if problem_type == 'grid_1d':
        return LIPETemporalLoss1D(config)
    elif problem_type == 'grid_2d':
        return LIPETemporalLoss2D(config)
    elif problem_type == 'pointcloud_3d':
        return LIPESpatialLoss(config)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


__all__ = [
    'LIPETemporalLoss1D',
    'LIPETemporalLoss2D',
    'LIPESpatialLoss',
    'build_lipe_loss'
]