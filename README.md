# UNOP: Physics-Constrained Unsupervised Neural Operator for Long-Horizon PDE Learning on Generalized Geometries

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

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/xinrrr0408/UNOP.git
cd UNOP

# Create conda environment
conda create -n unop python=3.9 -y
conda activate unop

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch_geometric
pip install -r requirements.txt


📂 Data Preparation
1D & 2D Problems: Please refer to the data generation code provided in the MNCP repository.
3D Problems: Please refer to the processed data provided in the Transolver repository.


python scripts/inference.py \
    --config configs/navier_stokes.yaml \
    --model checkpoints/model_best.pt