# utils/gaussian_rf.py
"""
Gaussian Random Field Generator (from MCNP)
"""

import torch
import math

class GaussianRF(object):
    def __init__(self, size, alpha=2.5, tau=7, sigma=None, boundary="periodic", device=None):
        self.dim = 2
        self.device = device
        self.size_val = size

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        
        k_max = size // 2
        
        wavenumbers = torch.cat((
            torch.arange(start=0, end=k_max, step=1, device=device),
            torch.arange(start=-k_max, end=0, step=1, device=device)
        ), 0).repeat(size, 1)

        k_x = wavenumbers.transpose(0, 1)
        k_y = wavenumbers

        self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
            (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0)
        )
        self.sqrt_eig[0, 0] = 0.0

        self.size = tuple([size for _ in range(self.dim)])

    def __call__(self, N):
        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real
    
    def sample(self, N):
        return self(N)