import numpy as np
import torch
from pathlib import Path
from .base_dataset import BasePhysicsDataset

class GridDataset(BasePhysicsDataset):
    def __init__(self, config):
        super().__init__(config['data'])
        self.full_config = config
        
        self.equation_type = self.config.get('equation_type', 'unknown')
        self.input_steps = self.config.get('input_steps', 0)
        self.pred_steps = self.config.get('pred_steps', 1)
        self.total_steps = self.input_steps + 1 + self.pred_steps
        
        self.return_full_trajectory = self.config.get('return_full_trajectory', False)
        
        self._load_data()
        
        print(f"\n{'='*70}\n📊 GridDataset 初始化\n{'='*70}")
        print(f"  方程: {self.equation_type}, 数据集: {self.split}")
        print(f"  样本数: {len(self)}, 空间尺寸: {self.get_spatial_size()}")
        print(f"  返回完整轨迹: {self.return_full_trajectory and self.split != 'train'}")
        print(f"{'='*70}\n")
    
    def _load_data(self):
        """通用数据加载 - 支持多种目录和文件名结构"""
        base_path = Path(self.data_root)
        
        # 优先使用通用的 subdir
        generic_subdir = self.config.get('subdir', None)
        if generic_subdir:
            base_path = base_path / generic_subdir
        else:
            subdir_n = self.config.get('subdir_n', None)
            if subdir_n:
                base_path = base_path / subdir_n
            
            subdir_e = self.config.get('subdir_e', None)
            if subdir_e:
                base_path = base_path / subdir_e

        print(f"🔍 正在搜索数据路径: {base_path}")
        
        if not base_path.exists():
            raise FileNotFoundError(f"路径不存在: {base_path}")
        

        all_files = list(base_path.iterdir())
        matching_files = []
        
        for f in all_files:
            if not f.is_file():
                continue
            
            filename_lower = f.name.lower()
            split_lower = self.split.lower()
            
            if (f'_{split_lower}' in filename_lower or 
                f'{split_lower}_' in filename_lower or
                f'-{split_lower}' in filename_lower or
                f'{split_lower}.' in filename_lower):
                matching_files.append(f)
        
        print(f"  目录中共 {len(all_files)} 个文件")
        if matching_files:
            print(f"   匹配到 {len(matching_files)} 个包含 '{self.split}' 的文件:")
            for f in matching_files:
                print(f"      - {f.name}")
        else:
            print(f"  未找到包含 '{self.split}' 的文件")
        
        loaded_data = None
        data_path = None
        
        for file_path in matching_files:
            print(f"\n尝试加载: {file_path.name}")
            loaded_data = self._try_load_file(file_path)
            
            if loaded_data is not None:
                data_path = file_path
                print(f"   加载成功！数据形状: {loaded_data.shape}")
                break
            else:
                print(f"   跳过此文件")
        
        if loaded_data is None:
            print(f"\n 未找到匹配文件，尝试旧的文件名格式...")
            loaded_data, data_path = self._load_legacy_format(base_path)
        
        # 最终检查
        if loaded_data is None or data_path is None:
            raise FileNotFoundError(
                f"   在 '{base_path}' 中未找到或无法加载任何有效数据文件！\n"
                f"   期望文件名包含: '{self.split}'\n"
                f"   目录中的文件: {[f.name for f in all_files if f.is_file()]}"
            )
        
        # 检查数据类型
        if not isinstance(loaded_data, np.ndarray):
            raise TypeError(f"  加载的数据不是numpy数组，而是: {type(loaded_data)}")
        
        print(f"\n  最终加载数据: {data_path.name}")
        print(f"   形状: {loaded_data.shape}, 类型: {loaded_data.dtype}")
        
        if loaded_data.ndim == 4 and self.full_config['model']['spatial_dim'] == 2:
            if loaded_data.shape[-1] in [1, 2, 3]:
                print(f"     转换维度: {loaded_data.shape} -> ", end="")
                loaded_data = loaded_data.transpose(0, 3, 1, 2)
                print(f"{loaded_data.shape}")
        
        self.data = loaded_data
        
        # 时间步检查
        if self.data.shape[1] < self.total_steps:
            raise ValueError(
                f"  数据时间步数不足！需要 {self.total_steps}, 实际 {self.data.shape[1]}"
            )
        
        self.T_total = self.data.shape[1]
        self.samples_per_trajectory = self.T_total - self.total_steps + 1
        self.num_trajectories = self.data.shape[0]
        
        print(f"   轨迹数: {self.num_trajectories}, 每条轨迹时间步: {self.T_total}")
        print(f"   样本数: {self.num_trajectories} 轨迹 × {self.samples_per_trajectory} 窗口")

    def __len__(self):
        if self.return_full_trajectory and self.split != 'train':
            return self.num_trajectories
        return self.num_trajectories * self.samples_per_trajectory

    def __getitem__(self, idx):
        spatial_dim = self.full_config['model']['spatial_dim']
        
        if self.return_full_trajectory and self.split != 'train':
            
            traj_idx = idx
            
            full_traj_raw = self.data[traj_idx]
            full_traj_raw = torch.from_numpy(full_traj_raw).float()
            
            
            if spatial_dim == 1:
                target_size = self.full_config['data']['grid_size'][0]
                T = full_traj_raw.shape[0]
                
                if full_traj_raw.shape[1] != target_size:
                    full_traj_reshaped = full_traj_raw.unsqueeze(1)
                    full_traj_aligned = torch.nn.functional.interpolate(
                        full_traj_reshaped, size=target_size, mode='linear', align_corners=True
                    ).squeeze(1)
                else:
                    full_traj_aligned = full_traj_raw
                
                full_traj = full_traj_aligned.unsqueeze(-1)  # [T, N, 1]
                
            elif self.equation_type == 'navier_stokes' and spatial_dim == 2:
                full_traj = full_traj_raw.unsqueeze(-1)  
            else:
                full_traj = full_traj_raw
            
            
            current = full_traj[0]  
            
            
            if self.input_steps > 0:
                history = current.unsqueeze(0).expand(self.input_steps, *current.shape).clone()
                
            else:
                history = torch.empty(0, *current.shape, dtype=current.dtype)
            
            target = full_traj[1] if full_traj.shape[0] > 1 else current
            
            return {
                'current': current,
                'history': history,
                'target': target,
                'spatial_size': self.get_spatial_size(),
                'full_trajectory': full_traj,
                'traj_idx': traj_idx,
            }
        else:
            traj_idx = idx // self.samples_per_trajectory
            time_idx = idx % self.samples_per_trajectory
            
            window_raw = self.data[traj_idx, time_idx:time_idx + self.total_steps]
            window_raw = torch.from_numpy(window_raw).float()
            
            if spatial_dim == 1:
                target_size = self.full_config['data']['grid_size'][0]
                if window_raw.shape[1] != target_size:
                    window_reshaped = window_raw.unsqueeze(1)
                    window_aligned = torch.nn.functional.interpolate(
                        window_reshaped, size=target_size, mode='linear', align_corners=True
                    ).squeeze(1)
                else:
                    window_aligned = window_raw
                window = window_aligned.unsqueeze(-1)
                
        
            elif self.equation_type == 'navier_stokes' and spatial_dim == 2:
                window = window_raw.unsqueeze(-1)
            else:
                window = window_raw
            
            if self.input_steps > 0:
                history = window[:self.input_steps]
            else:
                history = torch.empty(0, *window.shape[1:])
                
            current = window[self.input_steps]
            
            target_start_idx = self.input_steps + 1
            target = window[target_start_idx : target_start_idx + self.pred_steps]

            return {
                'current': current, 
                'history': history,
                'target': target.squeeze(0) if self.pred_steps == 1 else target,
                'spatial_size': self.get_spatial_size(),
            }

    def get_spatial_size(self):
        spatial_dim = self.full_config['model']['spatial_dim']
        if spatial_dim == 1:
            return (self.full_config['data']['grid_size'][0],)
        return self.data.shape[2:]
    
    def _load_legacy_format(self, base_path):
        """向后兼容：尝试旧的文件名格式"""
        possible_filenames = [
            f"data_{self.split}",
            f"data_{self.split}.npy",
            f"data_{self.split}.pt",
            f"{self.split}_data",
            f"{self.split}_data.npy",
            f"{self.split}_data.pt",
            f"{self.split}.npy",
            f"{self.split}.pt",
        ]
        
        subdir_e = self.config.get('subdir_e', None)
        if subdir_e:
            possible_filenames.extend([
                f"{subdir_e}_{self.split}_data",
                f"{subdir_e}_{self.split}_data.npy",
                f"{subdir_e}_{self.split}_data.pt",
            ])
        
        for filename in possible_filenames:
            path = base_path / filename
            if path.exists():
                print(f"   📥 尝试旧格式文件: {filename}")
                data = self._try_load_file(path)
                if data is not None:
                    return data, path
        
        return None, None
    
    def _try_load_file(self, file_path):
        """尝试用多种方式加载单个文件"""
        # 方法1: torch.load
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            data = torch.load(file_path, map_location=device)
            
            if isinstance(data, torch.Tensor):
                result = data.cpu().numpy()
                print(f"      torch.load ✓ (Tensor -> numpy)")
                return result
            
            elif isinstance(data, np.ndarray):
                print(f"      torch.load ✓ (numpy)")
                return data
            
            elif isinstance(data, dict):
                print(f"      torch.load ✓ (dict, keys: {list(data.keys())})")
                for key in ['data', 'x', 'input', 'u', 'samples']:
                    if key in data:
                        value = data[key]
                        if isinstance(value, torch.Tensor):
                            return value.cpu().numpy()
                        elif isinstance(value, np.ndarray):
                            return value
                
                for value in data.values():
                    if isinstance(value, torch.Tensor):
                        return value.cpu().numpy()
                    elif isinstance(value, np.ndarray):
                        return value
                
                print(f"         字典中未找到有效数据")
                return None
            
            else:
                print(f"         未知数据类型: {type(data)}")
                return None
                
        except Exception as e:
            print(f"      torch.load ✗ ({str(e)[:50]})")
        
        # 方法2: np.load
        try:
            data = np.load(file_path, allow_pickle=True)
            
            if isinstance(data, np.lib.npyio.NpzFile):
                keys = list(data.keys())
                print(f"      np.load ✓ (npz, keys: {keys})")
                return data[keys[0]]
            else:
                print(f"      np.load ✓")
                return data
                
        except Exception as e:
            print(f"      np.load ✗ ({str(e)[:50]})")
        
        # 方法3: pickle
        try:
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, torch.Tensor):
                print(f"      pickle.load ✓ (Tensor)")
                return data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                print(f"      pickle.load ✓ (numpy)")
                return data
            else:
                print(f"      pickle.load ✓ (但类型未知: {type(data)})")
                return None
                
        except Exception as e:
            print(f"      pickle.load ✗ ({str(e)[:50]})")
        
        return None


def grid_collate_fn(batch):
    """
    Collate 函数（支持完整轨迹模式）
    """
    current = torch.stack([item['current'] for item in batch])
    target = torch.stack([item['target'] for item in batch])
    
    if batch[0]['history'].shape[0] > 0:
        history = torch.stack([item['history'] for item in batch])
    else:
        history = None
    
    result = {
        'current': current, 
        'history': history, 
        'target': target,
        'spatial_size': batch[0]['spatial_size'],
    }
    
    if 'full_trajectory' in batch[0]:
        result['full_trajectory'] = torch.stack([item['full_trajectory'] for item in batch])
        result['traj_idx'] = torch.tensor([item['traj_idx'] for item in batch])
    
    return result