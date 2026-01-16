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
from utils import load_config
from data import GridDataset, grid_collate_fn
from torch.utils.data import DataLoader

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Running Inference on {device}")

    # =====================================================
    # 1. 加载配置和模型
    # =====================================================
    print(f"   Loading Config: {args.config}")
    config = load_config(args.config)
    
    # 修改 data split 为 test 以加载验证集
    config['data']['split'] = 'test' 

    print("  Building Model...")
    model = UNOP(config).to(device)

    print(f"  Loading Checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    # 处理可能的 'model_state_dict' 包装
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # =====================================================
    # 2. 准备数据 (从验证集取一个样本)
    # =====================================================
    print("  Loading Validation Dataset...")
    try:
        dataset = GridDataset(config)
        # 取第 0 个样本 (你可以修改 index 取其他样本)
        sample_idx = 0 
        sample = dataset[sample_idx] 
        
        # GridDataset 通常返回 {'u': ...} 或直接 Tensor
        if isinstance(sample, dict):
            # 获取完整轨迹 [T, H, W, C]
            gt_traj = sample.get('full_trajectory', sample.get('u'))
        else:
            gt_traj = sample
            
        # 增加 Batch 维度: [T, H, W, C] -> [1, T, H, W, C]
        gt_traj = gt_traj.unsqueeze(0).to(device).float()
        
    except Exception as e:
        print(f"  数据集加载失败: {e}")
        print("  提示：请检查 config 中的 data_path 是否正确指向了 .h5 或 .npy 文件")
        return

    # =====================================================
    # 3. 自回归推理 (Rollout)
    # =====================================================
    T_total = gt_traj.shape[1]
    steps_to_rollout = min(args.steps, T_total - 1)
    dt = config['physics']['delta_t']
    
    print(f"🔄 Running Rollout for {steps_to_rollout} steps...")
    
    # 初始条件: t=0
    current = gt_traj[:, 0] # [1, H, W, C]
    spatial_size = current.shape[1:-1] # (H, W)
    
    predictions = [current]
    
    with torch.no_grad():
        for t in range(steps_to_rollout):
            # 构造输入
            step_dt = torch.full((1,), dt, device=device)
            
            out = model({
                'current': current,
                'spatial_size': spatial_size,
                'target_time': step_dt
            })
            
            pred = out['output']
            predictions.append(pred)
            
            #   关键：把预测值作为下一步的输入 (Autoregressive)
            current = pred

    # 拼接结果 [T_pred, H, W, C] (去掉 Batch 维度)
    pred_traj = torch.cat(predictions, dim=0).cpu().numpy()
    gt_traj = gt_traj.squeeze(0).cpu().numpy()

    # =====================================================
    # 4. 可视化
    # =====================================================
    print(" Plotting results...")
    plot_2d_comparison(gt_traj, pred_traj, steps_to_rollout, args.save_path)


def plot_2d_comparison(gt, pred, steps, save_path):
    """
    画图函数：对比 GT, Pred, Error
    """
    # 选择要展示的时间步：0, T/4, T/2, 3T/4, T
    # 假设通道 0 是涡度 (Vorticity)
    channel = 0 
    
    indices = np.linspace(0, steps, 5, dtype=int)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    
    # 统一 Colorbar 范围
    vmin = gt[..., channel].min()
    vmax = gt[..., channel].max()
    
    cols = ['t={}'.format(i) for i in indices]
    rows = ['Ground Truth', 'Prediction', 'Abs Error']

    for i, idx in enumerate(indices):
        # 1. Ground Truth
        ax = axes[0, i]
        im1 = ax.imshow(gt[idx, ..., channel], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f't={idx}')
        if i == 0: ax.set_ylabel(rows[0], fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # 2. Prediction
        ax = axes[1, i]
        im2 = ax.imshow(pred[idx, ..., channel], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        if i == 0: ax.set_ylabel(rows[1], fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # 3. Error
        ax = axes[2, i]
        err = np.abs(gt[idx, ..., channel] - pred[idx, ..., channel])
        im3 = ax.imshow(err, cmap='jet', origin='lower') # Error 用不同的 colormap
        if i == 0: ax.set_ylabel(rows[2], fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        
        # 在最后一列加 Colorbar
        if i == 4:
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.suptitle(f'2D Navier-Stokes Rollout Results (Steps={steps})', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Result saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认使用你提供的路径
    parser.add_argument('--config', type=str, 
                        default='UNOP/configs/navier_stokes.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, 
                        default='UNOP/results/navier_stokes/model_best.pt',
                        help='Path to checkpoint')
    parser.add_argument('--steps', type=int, default=20, help='Number of rollout steps')
    parser.add_argument('--save_path', type=str, default='inference_result.png', help='Output image path')
    
    args = parser.parse_args()
    run_inference(args)