# lipe/temporal_integral_2d.py
"""
LIPE 2D Temporal Integral Loss (Navier-Stokes)
"""

import torch
import torch.nn.functional as F
import math


class LIPETemporalLoss2D:
    """
    2D Temporal Integral Loss (Navier-Stokes)
    """
    
    def __init__(self, config):
        self.config = config
    
    def __call__(self, omega_0, omega_1, config, aux_info=None):
        """
        Compute LIPE loss for 2D vorticity
        """
        device = omega_0.device
        B, H, W, C = omega_0.shape
        size = H
        
        # Physics parameters
        delta_t = getattr(config, 'delta_t', 0.1)
        nu = getattr(config, 'nu', 1e-4)
        k = getattr(config, 'k', 3)
        sup_w = getattr(config, 'sup_w', 2)
        sup_u = getattr(config, 'sup_u', 2)
        forcing_type = getattr(config, 'forcing_type', 'li')
        
        sup_size_w = sup_w * size
        sup_size_u = sup_u * size
        sigma = math.sqrt(2 * nu * delta_t)
        
        # Grid coordinates
        x = self._get_grid_coords(size, device)
        x = x.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Vorticity to velocity
        v_0 = self._w2v(omega_0, size)
        v_1 = self._w2v(omega_1, size)
        
        # Upsample velocity
        if sup_u > 1:
            v_0_sup = self._fft_upsample_2d(v_0, sup_u)
        else:
            v_0_sup = v_0
        
        # Characteristic line tracing (Heun method)
        mu_1 = x - v_1 * delta_t
        v_2 = self._sample_field_nearest(v_0_sup, mu_1, sup_size_u)
        mu = x - 0.5 * (v_1 + v_2) * delta_t
        
        # External forcing
        f = self._compute_forcing(x, mu, forcing_type)
        
        # Upsample vorticity
        if sup_w > 1:
            omega_0_sup = self._fft_upsample_2d(omega_0.squeeze(-1), sup_w)
        else:
            omega_0_sup = omega_0.squeeze(-1)
        
        # Integral sampling
        delta_grid, N_pts = self._get_sampling_grid(k, sup_size_w, device)
        loc_p = mu.unsqueeze(-2) + delta_grid.reshape(1, 1, 1, -1, 2)
        
        # Gaussian weights
        loc_p_discrete = (loc_p * sup_size_w).round() / sup_size_w
        dist_sq = ((loc_p_discrete - mu.unsqueeze(-2)) ** 2).sum(dim=-1)
        loc_density = (1 / (2 * math.pi * sigma**2)) * torch.exp(-dist_sq / (2 * sigma**2)) * (1 / sup_size_w**2)
        loc_density = loc_density / loc_density.detach().sum(dim=-1, keepdim=True)
        
        # Sample vorticity
        loc_p_periodic = loc_p_discrete - loc_p_discrete.floor()
        omega_vals = self._sample_multi_points(omega_0_sup, loc_p_periodic, sup_size_w)
        
        # Weighted sum + forcing
        omega_hat = (omega_vals * loc_density).sum(dim=-1) + f * delta_t
        
        # LIPE loss
        lipe_loss = torch.sqrt(torch.mean((omega_hat - omega_1.squeeze(-1)) ** 2))
        
        return lipe_loss, {'lipe_loss': lipe_loss.item()}
    
    def _w2v(self, w, size):
        """Vorticity to velocity conversion"""
        device = w.device
        N = size
        
        if w.dim() == 4:
            w = w.squeeze(-1)
        
        bw = w.shape[0]
        w0 = w.reshape(bw, N, N)
        k_max = N // 2
        
        w_h = torch.fft.rfft2(w0)
        
        k_y = torch.cat((
            torch.arange(start=0, end=k_max, step=1, device=device),
            torch.arange(start=-k_max, end=0, step=1, device=device)
        ), 0).repeat(N, 1).float()
        k_x = k_y.transpose(0, 1)
        
        k_x = k_x[..., :k_max + 1]
        k_y = k_y[..., :k_max + 1]
        
        lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap[0, 0] = 1.0
        
        psi_h = w_h / lap
        
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))
        
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))
        
        return torch.stack([q, v], dim=-1)
    
    def _fft_upsample_2d(self, x, factor):
        """FFT upsampling"""
        if factor == 1:
            return x
        
        has_channel = (x.dim() == 4)
        
        if has_channel:
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2)
        else:
            B, H, W = x.shape
            x = x.unsqueeze(1)
            C = 1
        
        new_H, new_W = H * factor, W * factor
        
        x_up = torch.fft.irfft(torch.fft.rfft(x, dim=2), dim=2, n=new_H)
        x_up = factor ** 2 * torch.fft.irfft(torch.fft.rfft(x_up, dim=3), dim=3, n=new_W)
        
        if has_channel:
            return x_up.permute(0, 2, 3, 1)
        else:
            return x_up.squeeze(1)
    
    def _sample_field_nearest(self, field, coords, sup_size):
        """Nearest neighbor sampling"""
        device = field.device
        B = field.shape[0]
        
        has_channel = (field.dim() == 4)
        if has_channel:
            C = field.shape[-1]
            results = []
            for c in range(C):
                results.append(self._sample_field_nearest(field[..., c], coords, sup_size))
            return torch.stack(results, dim=-1)
        
        H_out, W_out = coords.shape[1], coords.shape[2]
        
        index = coords - coords.floor()
        index = (index * sup_size).round().reshape(B, -1, 2).long()
        index = index.clone()
        index[index == sup_size] = 0
        
        N_points = H_out * W_out
        index_batch = torch.arange(B, device=device).reshape(-1, 1).repeat(1, N_points).reshape(-1).long()
        flat_index = index_batch * (sup_size ** 2) + index[:, :, 0].reshape(-1) * sup_size + index[:, :, 1].reshape(-1)
        
        field_flat = field.reshape(-1)
        sampled = torch.take(field_flat, flat_index).reshape(B, H_out, W_out)
        
        return sampled
    
    def _sample_multi_points(self, field, coords, sup_size):
        """Multi-point sampling"""
        device = field.device
        B, H, W, N_pts, _ = coords.shape
        
        loc_p = coords - coords.floor()
        loc_p = (loc_p * sup_size).round().reshape(B, -1, 2).long()
        loc_p = loc_p.clone()
        loc_p[loc_p == sup_size] = 0
        
        total_points = H * W * N_pts
        index_batch = torch.arange(B, device=device).reshape(-1, 1).repeat(1, total_points).reshape(-1).long()
        flat_index = index_batch * (sup_size ** 2) + loc_p[:, :, 0].reshape(-1) * sup_size + loc_p[:, :, 1].reshape(-1)
        
        field_flat = field.reshape(-1)
        sampled = torch.take(field_flat, flat_index).reshape(B, H, W, N_pts)
        
        return sampled
    
    def _get_grid_coords(self, size, device):
        """Generate grid coordinates"""
        gridx = torch.linspace(0, 1 - 1/size, size, device=device)
        gridx = gridx.reshape(size, 1, 1).repeat(1, size, 1)
        gridy = torch.linspace(0, 1 - 1/size, size, device=device)
        gridy = gridy.reshape(1, size, 1).repeat(size, 1, 1)
        
        return torch.cat([gridx, gridy], dim=-1)
    
    def _get_sampling_grid(self, k, sup_size_w, device):
        """Generate sampling grid"""
        delta_gridx = torch.linspace(-k/sup_size_w, k/sup_size_w, 2*k+1, device=device)
        delta_gridx = delta_gridx.reshape(-1, 1, 1).repeat(1, 2*k+1, 1)
        delta_gridy = torch.linspace(-k/sup_size_w, k/sup_size_w, 2*k+1, device=device)
        delta_gridy = delta_gridy.reshape(1, -1, 1).repeat(2*k+1, 1, 1)
        delta_grid = torch.cat([delta_gridx, delta_gridy], dim=-1)
        
        mask = delta_grid.norm(dim=-1) <= k / sup_size_w
        delta_grid = delta_grid[mask]
        N_pts = delta_grid.shape[0]
        
        return delta_grid, N_pts
    
    def _compute_forcing(self, x, mu, forcing_type):
        """Compute external forcing"""
        if forcing_type == 'li':
            f_1 = 0.1 * (torch.sin(2*math.pi*(x[..., 0] + x[..., 1])) + 
                         torch.cos(2*math.pi*(x[..., 0] + x[..., 1])))
            f_2 = 0.1 * (torch.sin(2*math.pi*(mu[..., 0] + mu[..., 1])) + 
                         torch.cos(2*math.pi*(mu[..., 0] + mu[..., 1])))
            return 0.5 * (f_1 + f_2)
        elif forcing_type == 'kolmogorov':
            f_1 = 0.1 * torch.cos(4 * math.pi * x[..., 0])
            f_2 = 0.1 * torch.cos(4 * math.pi * mu[..., 0])
            return 0.5 * (f_1 + f_2)
        else:
            return torch.zeros_like(x[..., 0])