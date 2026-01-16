# UNOP: Unsupervised Neural Operator for Physics Simulation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **UNOP** (Unsupervised Neural Operator), a unified framework for physics simulation that learns from physical laws without labeled data.

<p align="center">
  <img src="figures/framework.png" width="90%" alt="UNOP Framework"/>
</p>

## 🎯 Key Features

- **LIPE** (Latent Integral Physics Embedding): Unsupervised learning via stochastic integration, replacing differential residuals with integral constraints.
- **GALA** (Geometry-Agnostic Latent Adapter): Projects irregular geometries into regular latent space, unifying grids and point clouds.
- **GSEO** (Gated Spectral Evolution Operator): Spectral modeling with curvature-aware gating for long-horizon stability.

<!-- ## 📊 Supported Problems

| Problem | Dimension | Type | Equation |
|---------|-----------|------|----------|
| Convection-Diffusion | 1D | Transient | $\partial_t u + \beta \partial_x u = \kappa \partial_{xx} u$ |
| Navier-Stokes | 2D | Transient | $\partial_t \omega + (\mathbf{u} \cdot \nabla)\omega = \nu \nabla^2 \omega + f$ |
| Steady Flow | 3D | Steady-State | $\nabla \cdot \mathbf{u} = 0$, $(\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$ |

--- -->

## 🔧 Installation

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/UNOP.git
cd UNOP

# Create conda environment
conda create -n unop python=3.9 -y
conda activate unop

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements.txt

数据准备：
1D2D请参考MCNP的数据生成代码
3D的请参考Transover的processed_data

train_minimal.py --config configs/navier_stokes_2d_E1.yaml

python scripts/inference_ns_2d.py --model checkpoints/best_model.pt

 python scripts/inference.py --config configs/navier_stokes_2d_E1.yaml --model checkpoints/model_best_rollout.pt
