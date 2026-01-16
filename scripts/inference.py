#!/usr/bin/env python
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNOP
from utils import load_config, GaussianRF
from data import GridDataset

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Running Inference on {device}")


    if not os.path.exists(args.config):
        print(f" Config file not found: {args.config}")
        return

    config = load_config(args.config)
    config['data']['split'] = 'test' 

    model = UNOP(config).to(device)

    if not os.path.exists(args.model):
        print(f" Checkpoint file not found: {args.model}")
        return

    checkpoint = torch.load(args.model, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    print(" Preparing Data...")
    
    gt_traj = None
    has_gt = False
    
    try:
        dataset = GridDataset(config)
        sample = dataset[0] 
        if isinstance(sample, dict):
            gt_traj = sample.get('full_trajectory', sample.get('u'))
        else:
            gt_traj = sample
        
        gt_traj = gt_traj.unsqueeze(0).to(device).float()
        has_gt = True
        print("Mode: Validation (Dataset loaded)")
        
    except Exception as e:
        print(f"Dataset load failed: {e}")
        print("Mode: Demo (Random Generation)")
        has_gt = False
        return 

    if has_gt:
        dt = config['physics']['delta_t']
        T_total = gt_traj.shape[1]
        
        print("\n" + "="*50)
        print("Calculating Metrics...")
        print("="*50)
        tf_steps = min(10, T_total - 1)
        tf_errors = []
        
        with torch.no_grad():
            for t in range(tf_steps):
                current_gt = gt_traj[:, t]
                target_gt = gt_traj[:, t+1]
                
                spatial_size = current_gt.shape[1:-1]
                step_dt = torch.full((1,), dt, device=device)
                
                out = model({
                    'current': current_gt, 
                    'spatial_size': spatial_size,
                    'target_time': step_dt
                })
                pred = out['output']
                
                err = torch.norm(pred - target_gt) / (torch.norm(target_gt) + 1e-8)
                tf_errors.append(err.item())
        
        avg_tf_error = np.mean(tf_errors)
        print(f" [精度] TF Error (Mean 10-step):  {avg_tf_error * 100:.4f}%  <-- 也就是训练时的数值")


        rollout_steps = min(args.steps, T_total - 1)
        
        current = gt_traj[:, 0] 
        predictions = [current]
        
        with torch.no_grad():
            for t in range(rollout_steps):
                step_dt = torch.full((1,), dt, device=device)
                spatial_size = current.shape[1:-1]

                out = model({
                    'current': current,  
                    'spatial_size': spatial_size,
                    'target_time': step_dt
                })
                pred = out['output']
                predictions.append(pred)
                
                current = pred 
        

        pred_traj = torch.cat(predictions, dim=0).cpu().numpy()
        gt_traj_np = gt_traj.squeeze(0).cpu().numpy()
 
        final_err = np.linalg.norm(pred_traj[-1] - gt_traj_np[rollout_steps]) / \
                    (np.linalg.norm(gt_traj_np[rollout_steps]) + 1e-8)
        
        print(f"[推演] Rollout Error (Step {rollout_steps}): {final_err * 100:.4f}%  <-- 累积误差")
        print("="*50 + "\n")

    print("Plotting results...")
    if has_gt:
        plot_comparison(gt_traj_np, pred_traj, rollout_steps, args.save_path)

def plot_comparison(gt, pred, steps, save_path):
    """画对比图 (Jet colormap)"""
    indices = np.linspace(0, steps, 5, dtype=int)
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    
    channel = 0
    vmin, vmax = gt[..., channel].min(), gt[..., channel].max()
    rows = ['Ground Truth', 'Prediction', 'Abs Error']
    
    for i, idx in enumerate(indices):
        # GT
        im1 = axes[0, i].imshow(gt[idx, ..., channel], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, i].set_title(f't={idx}')
        
        # Pred
        im2 = axes[1, i].imshow(pred[idx, ..., channel], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        
        # Error
        err = np.abs(gt[idx, ..., channel] - pred[idx, ..., channel])
        im3 = axes[2, i].imshow(err, cmap='jet', origin='lower')
        
        if i == 0:
            for r in range(3): axes[r, 0].set_ylabel(rows[r], fontsize=14, fontweight='bold')
            
        if i == 4:
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.suptitle(f'Inference Results (Steps={steps})', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f" Result saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='inference_result.png')
    
    args = parser.parse_args()
    run_inference(args)