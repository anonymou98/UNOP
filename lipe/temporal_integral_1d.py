# lipe/temporal_integral_1d.py
"""
LIPE 1D Temporal Integral Loss (Convection-Diffusion)
"""

import torch
import torch.nn.functional as F
from lipe.utils import get_transition_matrix_1d

_MATRIX_CACHE = {}


class LIPETemporalLoss1D:
    """
    1D Temporal Integral Loss
    
    Implements Eq.5: u(x, t+Δt) = E_ξ[u(x - βΔt + ξ, t)]
    """
    
    def __init__(self, config):
        self.config = config
    
    def __call__(self, u0, u1, cfg, aux_info=None, **kwargs):
        """
        Compute LIPE loss
        """
        device = u0.device
        
        # Ensure [B, N] format
        if u0.dim() == 3:
            u_curr = u0.squeeze(-1)
        else:
            u_curr = u0
            
        if u1.dim() == 3:
            u_next_pred = u1.squeeze(-1)
        else:
            u_next_pred = u1

        # Handle size mismatch
        target_size = cfg.grid_size[0] if isinstance(cfg.grid_size, list) else cfg.grid_size
        
        if u_curr.shape[-1] != target_size:
            diff = u_curr.shape[-1] - target_size
            start = diff // 2
            u_curr = u_curr[:, start : start + target_size]
            u_next_pred = u_next_pred[:, start : start + target_size]
        
        # Physics parameters
        beta = getattr(cfg, 'b', 0.01)
        kappa = cfg.kappa
        dt = cfg.delta_t
        pad = getattr(cfg, 'pad', 5)

        # Get/cache transition matrix
        cache_key = (target_size, dt, kappa, beta, pad, device)
        if cache_key not in _MATRIX_CACHE:
            P = get_transition_matrix_1d(target_size, dt, kappa, beta, pad, device)
            _MATRIX_CACHE[cache_key] = P
        P = _MATRIX_CACHE[cache_key]

        # Compute integral target: I[u_curr] = u_curr @ P
        u_target = torch.matmul(u_curr, P)
        
        # LIPE loss
        lipe_loss = torch.sqrt(F.mse_loss(u_next_pred, u_target))

        return lipe_loss, {'lipe_loss': lipe_loss.item()}