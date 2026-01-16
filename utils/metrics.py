# utils/metrics.py
import torch
import torch.nn.functional as F
import numpy as np

class Metrics1D2D:
    @staticmethod
    def relative_l2(pred, gt):
        # [B, ...] -> [B, -1]
        pred_flat = pred.reshape(pred.shape[0], -1)
        gt_flat = gt.reshape(gt.shape[0], -1)
        diff = torch.norm(pred_flat - gt_flat, p=2, dim=-1)
        norm = torch.norm(gt_flat, p=2, dim=-1) + 1e-8
        return (diff / norm).mean().item()
    
    @staticmethod
    def relative_linf(pred, gt):
        pred_flat = pred.reshape(pred.shape[0], -1)
        gt_flat = gt.reshape(gt.shape[0], -1)
        diff = torch.max(torch.abs(pred_flat - gt_flat), dim=-1).values
        norm = torch.max(torch.abs(gt_flat), dim=-1).values + 1e-8
        return (diff / norm).mean().item()
    
    @staticmethod
    def mse(pred, gt):
        return F.mse_loss(pred, gt).item()
    
    @staticmethod
    def rmse(pred, gt):
        return torch.sqrt(F.mse_loss(pred, gt)).item()

def compute_all_metrics_1d2d(pred, gt):
    return {
        'rel_l2': Metrics1D2D.relative_l2(pred, gt),
        'rel_linf': Metrics1D2D.relative_linf(pred, gt),
        'mse': Metrics1D2D.mse(pred, gt),
        'rmse': Metrics1D2D.rmse(pred, gt),
    }

class Metrics3D:
    @staticmethod
    def relative_l2_velocity(pred, gt):
        # pred, gt shape: [N, 3] or [B, N, 3]
        diff = torch.norm(pred - gt, p=2)
        norm = torch.norm(gt, p=2) + 1e-8
        return (diff / norm).item()
    
    @staticmethod
    def rmse_velocity(pred, gt):
        return torch.sqrt(F.mse_loss(pred, gt)).item()
    
    @staticmethod
    def max_error(pred, gt):
        return torch.max(torch.abs(pred - gt)).item()

def compute_all_metrics_3d(pred, gt, coords=None, surf_mask=None):
    metrics = {
        'rel_l2_velo': Metrics3D.relative_l2_velocity(pred[...,:3], gt[...,:3]),
        'rmse': Metrics3D.rmse_velocity(pred[...,:3], gt[...,:3]),
        'max_error': Metrics3D.max_error(pred, gt)
    }
    return metrics

class TestMetricsCollector:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {}
        self.count = 0
        
    def update(self, metric_dict):
        for k, v in metric_dict.items():
            self.metrics[k] = self.metrics.get(k, 0.0) + v
        self.count += 1
        
    def compute(self):
        if self.count == 0: return {}
        return {k: v / self.count for k, v in self.metrics.items()}


@torch.no_grad()
def validate_teacher_forcing(model, dataloader, device, dt):
    """
    Teacher Forcing: Input GT[t] -> Predict [t+1] -> Compare with GT[t+1]
    """
    model.eval()
    errors = []
    
    for batch in dataloader:
        if isinstance(batch, dict):
            gt_traj = batch.get('full_trajectory', batch.get('u', list(batch.values())[0]))
        else:
            gt_traj = batch

        if gt_traj is None: continue

        gt_traj = gt_traj.to(device).float()
        # [B, T, Spatial..., C]
        B, T = gt_traj.shape[0], gt_traj.shape[1]
        
        steps_to_check = min(T - 1, 10) 
        
        for t in range(steps_to_check):
            u_current = gt_traj[:, t]   # Input is GT
            u_next_gt = gt_traj[:, t+1] # Target
            
            spatial_size = u_current.shape[1:-1]
            
            target_time = torch.full((B,), dt, device=device)
            
            out = model({
                'current': u_current,
                'spatial_size': spatial_size,
                'target_time': target_time
            })
            u_next_pred = out['output']
            
            err = Metrics1D2D.relative_l2(u_next_pred, u_next_gt)
            errors.append(err)
            
    return np.mean(errors) if errors else float('inf')

@torch.no_grad()
def validate_rollout(model, dataloader, device, dt, max_steps=100):
    """
    Autoregressive Rollout: Input Pred[t] -> Predict [t+1]
    """
    model.eval()
    final_step_errors = []
    trajectories_analyzed = 0
    
    for batch in dataloader:
        if isinstance(batch, dict):
            gt_traj = batch.get('full_trajectory', batch.get('u', list(batch.values())[0]))
        else:
            gt_traj = batch
            
        if gt_traj is None: continue

        gt_traj = gt_traj.to(device).float()
        B, T_data = gt_traj.shape[0], gt_traj.shape[1]
        
        steps_to_run = min(T_data - 1, max_steps)
        if steps_to_run < 1: continue

        current = gt_traj[:, 0]
        
        spatial_size = current.shape[1:-1]
        
        target_time = torch.full((B,), dt, device=device)

        for t in range(steps_to_run):
            out = model({
                'current': current,
                'spatial_size': spatial_size,
                'target_time': target_time
            })
            pred = out['output']
            current = pred 
        
        gt_final = gt_traj[:, steps_to_run]
        err = Metrics1D2D.relative_l2(current, gt_final)
        
        final_step_errors.append(err)
        trajectories_analyzed += B
        
    return (np.mean(final_step_errors) if final_step_errors else float('inf')), trajectories_analyzed