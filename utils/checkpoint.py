# utils/checkpoint.py
"""
Checkpoint Manager
"""

import torch
from pathlib import Path
import shutil


class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False, **kwargs):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            **kwargs
        }
        
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(state, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            shutil.copy2(latest_path, best_path)
            
        if epoch % 10 == 0:
            backup_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            shutil.copy2(latest_path, backup_path)
            
        self._cleanup_old_checkpoints()
            
    def load_checkpoint(self, name='latest', model=None, optimizer=None, scheduler=None):
        if name == 'latest':
            path = self.checkpoint_dir / 'checkpoint_latest.pth'
        elif name == 'best':
            path = self.checkpoint_dir / 'checkpoint_best.pth'
        else:
            path = self.checkpoint_dir / f'{name}.pth'
            
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        checkpoint = torch.load(path, map_location='cpu')
        
        if model:
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        backups = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        if len(backups) > self.keep_last_n:
            for old in backups[:-self.keep_last_n]:
                old.unlink()