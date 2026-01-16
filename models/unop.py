# models/unop.py
"""
UNOP: Unsupervised Neural Operator for Physics Simulation

Architecture:
    Input → GALA (Encoder) → GSEO (Evolution) → Decoder → Output
    
Training:
    LIPE (Latent Integral Physics Embedding) - Unsupervised
"""

import torch
import torch.nn as nn
import traceback

from .gala import GALAGridEncoder, GALAGraphEncoder, GALAProjection
from .decoder import GridDecoder, PointCloudSampler
from .gseo import GSEOEvolution, GSEOEvolution3D
from operators.coordinate_utils import CoordinateNormalizer


class UNOP(nn.Module):
    """
    Unsupervised Neural Operator for Physics Simulation
    
    Components:
        - GALA: Geometry-Agnostic Latent Adapter
        - GSEO: Gated Spectral Evolution Operator
        - Decoder: Grid/PointCloud decoding
    
    Training:
        - LIPE: Latent Integral Physics Embedding (unsupervised)
    """
    
    def __init__(self, config):
        super().__init__()
        
        model_config = config['model']
        self.config = model_config
        self.physics_config = config.get('physics', {})
        self.problem_type = model_config['problem_type']
        self.spatial_dim = model_config['spatial_dim']
        self.embed_dim = model_config['embed_dim']
        self.in_channels = model_config['in_channels']
        self.out_channels = model_config['out_channels']
        self.data_config = config['data']
        
        # 缓存变量初始化
        self._cached_grid_coords = None
        self._cached_grid_size = None
        self._cached_device = None
       
        print("\n" + "="*70)
        print(f"  Building UNOP: {self.problem_type} ({self.spatial_dim}D)")
        print("="*70)

        # ================================================================
        # 1. GALA: Geometry-Agnostic Latent Adapter
        # ================================================================
        self.gala_encoder = self._build_gala_encoder()
        
        # ================================================================
        # 2. GSEO: Gated Spectral Evolution Operator
        # ================================================================
        gseo_config = self.config.get('gseo', self.config.get('afno', {})).copy()
        gseo_config['history_len'] = self.data_config.get('input_steps', 5)
        gseo_config['boundary_type'] = self.physics_config.get('boundary_type', 'periodic')

        if self.problem_type == 'pointcloud_3d':
            gseo_config['mode'] = 'steady'
            self.gseo = GSEOEvolution3D(
                spatial_dim=self.spatial_dim,
                embed_dim=self.embed_dim,
                **gseo_config
            )
            self.gseo_evolution = None
            
            # 预设网格尺寸用于缓存
            self._cached_grid_size = tuple(self.config['grid_size'])
        else:
            gseo_config['mode'] = 'transient'
            self.gseo_evolution = GSEOEvolution(
                embed_dim=self.embed_dim,
                spatial_dim=self.spatial_dim,
                **gseo_config
            )
            self.gseo = None
            
        print(f"  [GSEO] Gated Spectral Evolution Operator")

        # ================================================================
        # 3. Decoder
        # ================================================================
        self.decoder = self._build_decoder()
        
        self.register_buffer('output_scale', torch.tensor(1.0))
        
        if self.problem_type == 'pointcloud_3d':
            self.coord_normalizer = CoordinateNormalizer()
        else:
            self.coord_normalizer = None

        # 消融实验支持
        self.ablation_config = {
            'use_gala': True,
            'use_gate': True,
            'use_spectral': True,
        }

        # 纯点云 baseline（w/o GALA 时使用）
        self.point_only_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.out_channels),
        )
        
        self.high_freq_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.out_channels),
        )
        
        # 初始化为 0
        nn.init.zeros_(self.high_freq_head[-1].weight)
        nn.init.zeros_(self.high_freq_head[-1].bias)
        
        self._print_model_info()
    
    def _build_gala_encoder(self):
        """Build GALA encoder"""
        if self.problem_type.startswith('grid'):
            print(f"  [GALA] GridEncoder ({self.spatial_dim}D)")
            grid_size = self.data_config.get('grid_size')
            encoder_config = self.config.get('gala', self.config.get('encoder', {}))
            
            return GALAGridEncoder(
                spatial_dim=self.spatial_dim,
                in_channels=self.in_channels,
                embed_dim=self.embed_dim,
                grid_size=grid_size,
                **encoder_config
            )
        
        elif self.problem_type == 'pointcloud_3d':
            print(f"  [GALA] GraphEncoder + Projection")
            grapher_config = self.config.get('gala', self.config.get('grapher', {})).copy()
            grapher_config['feature_width'] = self.embed_dim
            grapher_config['input_features'] = self.in_channels - self.spatial_dim
            grapher_config['output_features'] = self.embed_dim
            grapher_config['pos_dim'] = self.spatial_dim
            
            print(f"      GraphEncoder config: input_features={grapher_config['input_features']}, "
                f"pos_dim={grapher_config['pos_dim']}, "
                f"feature_width={grapher_config['feature_width']}")
            
            graph_encoder = GALAGraphEncoder(**grapher_config)
            
            projection = GALAProjection(
                point_dim=self.embed_dim,
                grid_dim=self.embed_dim,
                grid_size=tuple(self.config['grid_size']),
                k_neighbors=self.config.get('k_neighbors_proj', 32)
            )
            
            return nn.ModuleDict({
                'graph_encoder': graph_encoder,
                'projection': projection
            })
        else:
            raise ValueError(f"Unknown problem_type for GALA: {self.problem_type}")
    
    def set_ablation(self, use_gala=True, use_gate=True, use_spectral=True):
        """设置消融配置"""
        self.ablation_config = {
            'use_gala': use_gala,
            'use_gate': use_gate,
            'use_spectral': use_spectral,
        }
        print(f"✅ Ablation: GALA={use_gala}, Gate={use_gate}, Spectral={use_spectral}")
    
    def _build_decoder(self):
        """Build decoder"""
        if self.problem_type.startswith('grid'):
            print(f"  [Decoder] GridDecoder")
            return GridDecoder(
                embed_dim=self.embed_dim,
                out_channels=self.out_channels,
                spatial_dim=self.spatial_dim,
                **self.config.get('decoder', {})
            )
        
        elif self.problem_type == 'pointcloud_3d':
            print(f"  [Decoder] PointCloudSampler")
            return PointCloudSampler(
                grid_dim=self.embed_dim,
                out_dim=self.out_channels,
                spatial_dim=3,
                **self.config.get('decoder', {})
            )
        else:
            raise ValueError(f"Unknown problem_type for decoder: {self.problem_type}")
    
    def forward(self, data):
        """Unified forward pass"""
        if self.problem_type.startswith('grid'):
            return self._forward_grid(data)
        elif self.problem_type == 'pointcloud_3d':
            return self._forward_pointcloud(data)
        else:
            raise ValueError(f"Unknown forward path for problem_type: {self.problem_type}")
    
    def _forward_grid(self, data):
        """1D/2D transient: use GSEO Evolution"""
        current = data['current']
        dt = data.get('target_time')
        spatial_size = data['spatial_size']
        B = current.shape[0]
        
        if dt is None:
            dt = torch.ones(B, device=current.device)
        
        if self.spatial_dim == 1:
            inp = current.permute(0, 2, 1)
        elif self.spatial_dim == 2:
            inp = current.permute(0, 3, 1, 2)
        
        # GALA: Encode
        state_z = self.gala_encoder(inp, spatial_size)
        
        # GSEO: Evolve
        delta_z, aux_info = self.gseo_evolution(
            z=state_z,
            dt=dt,
            physical_field=current.detach(),
            spatial_size=spatial_size
        )
        
        evolved_z = state_z + delta_z
        
        # Decode
        prediction = self.decoder(evolved_z, spatial_size)
        
        return {'output': prediction, 'aux_info': aux_info}

    def _get_cached_grid_coords(self, D, H, W, device):
        """Get cached grid coordinates"""
        current_size = (D, H, W)
        
        if (self._cached_grid_coords is not None and 
            self._cached_grid_size == current_size and 
            self._cached_device == device):
            return self._cached_grid_coords
        
        z = torch.linspace(0, 1, D, device=device)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        self._cached_grid_coords = coords
        self._cached_grid_size = current_size
        self._cached_device = device
        
        return coords

    def _project_curvature_to_grid(self, point_curvature, coords, spatial_size, device):
        """Project point curvature to grid"""
        from torch_geometric.nn import knn
        
        D, H, W = spatial_size
        N_grid = D * H * W
        N_points = coords.shape[0]
        
        grid_coords = self._get_cached_grid_coords(D, H, W, device)
        
        k = min(4, N_points)
        
        assign_index = knn(coords, grid_coords, k)
        grid_idx = assign_index[0]
        point_idx = assign_index[1]
        
        relative_pos = grid_coords[grid_idx] - coords[point_idx]
        distances = torch.norm(relative_pos, dim=-1) + 1e-6
        weights = 1.0 / distances
        weights = weights.view(N_grid, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        neighbor_curvs = point_curvature[point_idx, 0].view(N_grid, k)
        grid_curvature = (neighbor_curvs * weights).sum(dim=-1, keepdim=True)
        
        return grid_curvature.unsqueeze(0)

    def _forward_pointcloud(self, data):
        """3D steady-state point cloud"""
        
        # 1. GALA: Encode
        point_features = self.gala_encoder['graph_encoder'](data)
        coords = data.pos
        
        if self.coord_normalizer is not None:
            coords = self.coord_normalizer.normalize(coords)
        
        # Ablation: w/o GALA
        if not self.ablation_config.get('use_gala', True):
            output = self.point_only_head(point_features)
            return {'output': output, 'aux_info': {'mode': 'no_gala'}}
        
        # 2. GALA: Project to grid
        grid_features = self.gala_encoder['projection'](point_features, coords)
        B, D, H, W, C = grid_features.shape
        spatial_size = (D, H, W)
        
        # 3. Process geometry curvature
        geometry_curvature_grid = None
        point_curvature = getattr(data, 'curvature', None)
        if point_curvature is not None:
            geometry_curvature_grid = self._project_curvature_to_grid(
                point_curvature, coords, spatial_size, grid_features.device
            )
        
        # 4. GSEO: Refine
        grid_flat = grid_features.reshape(B, -1, C)
        
        refined_features, aux_info = self.gseo(
            current_features=grid_flat,
            physical_field=grid_features,
            geometry_curvature=geometry_curvature_grid,
            spatial_size=spatial_size,
            return_aux=True
        )
        
        processed_grid = refined_features.reshape(B, D, H, W, C)
        
        # 5. Decode
        grid_coords = self._get_cached_grid_coords(D, H, W, processed_grid.device)
        grid_physics = self.decoder(processed_grid, grid_coords)
        grid_physics = grid_physics.reshape(B, D, H, W, self.out_channels)
        
        # Grid base + Point residual
        output_grid = self.decoder(processed_grid, coords)
        output_point = self.high_freq_head(point_features)
        
        output = output_grid + output_point
        
        return {
            'output': output,
            'aux_info': aux_info,
            'grid_features': processed_grid,
            'grid_physics': grid_physics,
            'coords': coords,
        }

    @staticmethod
    def _create_grid_coords(spatial_size, device):
        """Create regular grid coordinates (for 3D)"""
        if len(spatial_size) != 3:
            raise ValueError("This helper is intended for 3D grid coordinates.")
        D, H, W = spatial_size
        z = torch.linspace(0, 1, D, device=device)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        return coords

    def fit_normalizer(self, dataset):
        """Fit coordinate normalizer from dataset"""
        if self.coord_normalizer is None:
            return
        if self.problem_type != 'pointcloud_3d':
            return
            
        print("INFO: Fitting coordinate normalizer...")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
        
        sample_indices = torch.randperm(len(dataset))[:min(100, len(dataset))]
        
        coords_list = []
        for i in sample_indices:
            try:
                data_seq = dataset[i]
                if data_seq and hasattr(data_seq[0], 'pos'):
                    coords_list.append(data_seq[0].pos)
            except Exception as e:
                print(f"WARN: Failed to sample data (index {i}): {e}")
                traceback.print_exc()

        if not coords_list:
            raise ValueError("No coordinates collected from dataset!")
            
        all_coords = torch.cat(coords_list, dim=0)
        self.coord_normalizer.fit(all_coords)
        
        print(f"INFO: Coordinate normalizer fitted (based on {all_coords.shape[0]} points).")
        
    def _print_model_info(self):
        """Print model parameter statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total Parameters:     {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print("="*70)