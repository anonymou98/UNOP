#!/usr/bin/env python
# scripts/train_minimal.py

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNOP
from data import GridDataset, grid_collate_fn
from lipe import build_lipe_loss
from utils import load_config, GaussianRF
from utils.metrics import validate_teacher_forcing, validate_rollout

def setup_seed(seed=42):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ==============================================================================
# 🔥 新增：课程学习策略函数
# ==============================================================================
def get_progressive_rollout_steps(epoch, total_epochs, max_rollout_steps, config):
    """
    根据训练进度动态调整 Rollout 步数
    Epoch 0% - 20%: 1 步
    Epoch 20% - 50%: 线性增加到 max_steps
    Epoch > 50%: 保持 max_steps
    """
    # 如果配置里没开 rollout，或者最大步数是1，直接返回1
    if not config['training'].get('use_rollout', False):
        return 1
    
    # 简单的课程策略
    if epoch < total_epochs * 0.2:
        return 1
    elif epoch < total_epochs * 0.5:
        ratio = (epoch - total_epochs * 0.2) / (total_epochs * 0.3)
        steps = 1 + int((max_rollout_steps - 1) * ratio)
        return min(steps, max_rollout_steps)
    else:
        return max_rollout_steps

# ==============================================================================
# 数据生成函数
# ==============================================================================
def generate_ic_1d(batch_size, grid_size, config, device):
    N = config.get('physics', {}).get('N', 5)
    FINE_GRID = 1024
    sub_x = FINE_GRID // grid_size
    B_n = torch.rand(batch_size, N, device=device).reshape(batch_size, N, 1) * (2.0/N)
    x_fine = torch.linspace(0, 1 - 1/FINE_GRID, FINE_GRID, device=device).reshape(1, 1, -1)
    n = 2 * torch.arange(1, N+1, device=device).float().reshape(1, -1, 1)
    sin_grid = torch.sin(torch.pi * n * x_fine)
    u0_fine = (B_n * sin_grid).sum(dim=1).unsqueeze(-1)
    u0 = u0_fine[:, ::sub_x, :]
    return u0

def generate_ic_2d(batch_size, grid_size, grf_generator, device):
    if grf_generator is not None:
        w_0 = grf_generator(batch_size).unsqueeze(-1)
    else:
        grf = GaussianRF(grid_size, device=device)
        w_0 = grf(batch_size).unsqueeze(-1)
    return w_0

# ==============================================================================
# 2. 训练主循环 (含课程学习)
# ==============================================================================
def train_grid(model, config, device, epochs=100, val_loader=None):
    print("\n" + "="*60)
    print("  UNOP Grid Training (With Curriculum Learning)")
    print("="*60)
    
    spatial_dim = config['model']['spatial_dim']
    batch_size = config['training']['batch_size']
    grid_size = config['data']['grid_size'][0]
    dt = config['physics']['delta_t']
    
    loss_fn = build_lipe_loss(config['model']['problem_type'], config)
    physics_cfg = SimpleNamespace(**config.get('physics', {}))
    physics_cfg.grid_size = config['data']['grid_size']
    
    optimizer = AdamW(model.parameters(), lr=config['training']['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    grf_generator = None
    if spatial_dim == 2:
        grf_generator = GaussianRF(size=grid_size, device=device)
    
    steps_per_epoch = config['training'].get('steps_per_epoch', 100)
    val_interval = config['training'].get('val_interval', 10)
    
    # 🔥 获取最大 Rollout 步数配置
    max_rollout_steps = config['training'].get('rollout_steps', 1)
    
    best_rollout_error = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # 🔥 计算当前 Epoch 应该 Rollout 多少步
        current_rollout_steps = get_progressive_rollout_steps(epoch, epochs, max_rollout_steps, config)
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch:03d} [Rollout={current_rollout_steps}]", leave=False)
        
        for step in pbar:
            optimizer.zero_grad()
            
            # 1. 生成初始状态
            if spatial_dim == 1:
                state = generate_ic_1d(batch_size, grid_size, config, device)
                spatial_size = (grid_size,)
            else:
                state = generate_ic_2d(batch_size, grid_size, grf_generator, device)
                spatial_size = (grid_size, grid_size)
            
            # 🔥 课程学习循环
            batch_loss = 0.0
            step_dt_tensor = torch.full((batch_size,), dt, device=device)
            
            # 循环推演
            for k in range(current_rollout_steps):
                # 前向预测
                out = model({
                    'current': state,
                    'spatial_size': spatial_size,
                    'target_time': step_dt_tensor
                })
                pred = out['output']
                
                # 计算物理损失 (Loss based on transition from state -> pred)
                step_loss, _ = loss_fn(state, pred, physics_cfg, out.get('aux_info'))
                
                # 累积损失 (平均化)
                batch_loss += step_loss / current_rollout_steps
                
                # 🔥 关键：更新状态用于下一步，并切断反向传播图防止显存爆炸
                # 但保留 pred 用于当前步的梯度计算
                if k < current_rollout_steps - 1:
                    state = pred.detach() 

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            pbar.set_postfix({'loss': f'{batch_loss.item():.4e}'})
        
        scheduler.step()
        avg_train_loss = epoch_loss / steps_per_epoch

        # --- 验证阶段 ---
        if epoch % val_interval == 0 and val_loader is not None:
            print(f"\nEvaluating at Epoch {epoch}...")
            tf_error = validate_teacher_forcing(model, val_loader, device, dt)
            rollout_err, _ = validate_rollout(model, val_loader, device, dt, max_steps=20)
            
            print(f"  [Train] Physics Loss: {avg_train_loss:.4e}")
            print(f"  [Val]   TF Error (Mean 10-step): {tf_error*100:.4f}%")
            print(f"  [Val]   Rollout Error (T=20):    {rollout_err*100:.4f}%")
            
            if rollout_err < best_rollout_error:
                best_rollout_error = rollout_err
                save_path = Path(config['training']['save_dir']) / 'model_best_rollout.pt'
                torch.save(model.state_dict(), save_path)
                print(f"  🏆 New Best Model saved!")
        else:
            print(f"Epoch {epoch:03d}: Loss = {avg_train_loss:.4e}")

    print("\n✅ Training complete!")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    if args.epochs: config['training']['epochs'] = args.epochs
    
    save_dir = Path(config['training'].get('save_dir', './results'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("📂 Preparing Validation Dataset...")
    try:
        val_config = config.copy()
        val_config['data']['split'] = 'test' 
        val_ds = GridDataset(val_config)
        val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], 
                                collate_fn=grid_collate_fn, shuffle=False)
        print(f"   Validation samples: {len(val_ds)}")
    except Exception as e:
        print(f"⚠️ Could not load validation dataset: {e}")
        val_loader = None
    
    model = UNOP(config).to(device)
    
    train_grid(model, config, device, 
               epochs=config['training']['epochs'], 
               val_loader=val_loader)
    
    if val_loader is not None:
        print("\n" + "="*60)
        print("🚀 Running Final Long-term Inference Test (T=50)")
        print("="*60)
        
        best_model_path = save_dir / 'model_best_rollout.pt'
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
            print("✅ Loaded best model for testing.")
            
        dt = config['physics']['delta_t']
        long_err, count = validate_rollout(model, val_loader, device, dt, max_steps=50)
        
        print(f"Final Test Results (Autoregressive):")
        print(f"  Tested on {count} trajectories.")
        print(f"  Error at T=50: {long_err*100:.4f}%")
        print("="*60)

if __name__ == '__main__':
    main()