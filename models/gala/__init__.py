# models/gala/__init__.py
"""
GALA: Geometry-Agnostic Latent Adapter

Projects irregular geometries into regular latent space,
decoupling geometry from computation and unifying grids and point clouds.
"""

from .grid_encoder import GALAGridEncoder
from .graph_encoder import GALAGraphEncoder
from .projection import GALAProjection
from .mlp import MLP

__all__ = [
    'GALAGridEncoder',
    'GALAGraphEncoder',
    'GALAProjection',
    'MLP',
]