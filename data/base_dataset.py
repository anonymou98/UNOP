# ============================================================================
# data/base_dataset.py
# 数据集基类
# ============================================================================
"""
定义统一的数据集接口
"""

from torch.utils.data import Dataset
import torch


class BasePhysicsDataset(Dataset):
    """
    物理数据集基类
    
    所有数据集应该实现：
    - __len__(): 返回样本数
    - __getitem__(idx): 返回一个样本
    - get_spatial_size(): 返回空间尺寸
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_root = config.get('root', './data')
        self.split = config.get('split', 'train')
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def get_spatial_size(self):
        """返回空间尺寸 (N,) 或 (H, W) 或 (D, H, W)"""
        raise NotImplementedError
    
    def get_statistics(self):
        """返回数据集统计信息（可选）"""
        return {
            'num_samples': len(self),
            'split': self.split,
        }