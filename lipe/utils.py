# lipe/utils.py
"""
LIPE Utilities: Transition matrix and helpers
"""

import torch
import numpy as np
import math


def get_transition_matrix_1d(grid_size, dt, kappa, beta, pad=3, device='cpu'):
    """
    Compute 1D transition matrix for convection-diffusion
    (Used by LIPETemporalLoss1D)
    
    Implements: u(x, t+dt) = sum_i P_ij * u(x_i, t)
    
    Args:
        grid_size: Number of grid points
        dt: Time step
        kappa: Diffusion coefficient
        beta: Advection velocity
        pad: Padding for integration window
        device: Torch device
    
    Returns:
        P: [N, N] transition matrix
    """
    # Compute safe padding
    sigma = math.sqrt(2 * kappa * dt) if kappa > 0 else 0
    advection_shift = abs(beta * dt)
    
    needed_range = advection_shift + 4 * sigma
    dx = 1.0 / grid_size
    
    safe_pad = max(pad, int(math.ceil(needed_range / dx)) + 2)
    
    # Extended source grid
    x_extended = torch.linspace(
        -safe_pad, 1 + safe_pad - 1/grid_size, 
        (2*safe_pad + 1) * grid_size, 
        device=device
    ).reshape(-1, 1)
    
    # Target grid with advection
    mu_base = torch.linspace(0, 1 - 1/grid_size, grid_size, device=device)
    shift = beta * dt
    mu = mu_base + shift
    
    # Periodic wrapping
    mu = mu - torch.floor(mu)
    mu = mu.reshape(1, -1)
    
    # Gaussian weights
    effective_sigma = max(sigma, 1e-8)
    dist_sq = (x_extended - mu) ** 2
    W = (1.0 / (math.sqrt(2 * math.pi) * effective_sigma)) * torch.exp(-dist_sq / (2 * effective_sigma**2))
    W = W / grid_size
    
    # Fold for periodicity
    W_reshaped = W.reshape(2*safe_pad + 1, grid_size, grid_size)
    P = W_reshaped.sum(dim=0)
    
    return P

