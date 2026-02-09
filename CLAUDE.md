# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Official implementation of "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023). Optimizes 3D Gaussians from Structure-from-Motion inputs for real-time novel-view synthesis. Non-commercial license.

## Common Commands

### Environment Setup
```bash
git submodule update --init --recursive  # If cloned without --recursive
conda env create --file environment.yml
conda activate gaussian_splatting
```

### Training
```bash
python train.py -s <path_to_dataset>                    # Basic training
python train.py -s <dataset> --eval                      # With train/test split
python train.py -s <dataset> -m <output_path>            # Custom output directory
python train.py -s <dataset> --optimizer_type sparse_adam # Fast training (~2.7× speedup)
```

### Rendering & Evaluation
```bash
python render.py -m <model_path>          # Generate renderings from trained model
python metrics.py -m <model_path>         # Compute PSNR, SSIM, LPIPS
```

### Dataset Conversion (your own images)
```bash
python convert.py -s <location>            # Runs COLMAP + undistortion
python convert.py -s <location> --resize    # Also create 1/2, 1/4, 1/8 resolution copies
python convert.py -s <location> --skip_matching  # If COLMAP data already exists
```
Requires COLMAP (and ImageMagick for `--resize`) on system PATH. Input images go in `<location>/input/`.

### Full Benchmark Evaluation
```bash
python full_eval.py -m360 <mipnerf360> -tat <tanks_temples> -db <deepblending>
```

### Building SIBR Viewers (C++, Linux)
```bash
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
```

## Architecture

### Training Pipeline (`train.py`)
1. Loads dataset (COLMAP or Blender NeRF format) via `scene/dataset_readers.py`
2. Initializes `GaussianModel` from SfM point cloud
3. Iterates 30k steps: render → loss → backward → densify/prune → optimizer step
4. Loss: `(1-λ)·L1 + λ·(1-SSIM)` where λ=0.2, plus optional depth regularization

### Core Components

- **`scene/gaussian_model.py`** — Central model class. Each Gaussian has: position (`_xyz`), scaling (`_scaling`), rotation quaternion (`_rotation`), opacity (`_opacity`), spherical harmonic coefficients (`_features_dc`, `_features_rest`), and optional per-image exposure (`_exposure`). Handles parameter setup, optimization, densification (split/clone/prune), and serialization to PLY.

- **`gaussian_renderer/__init__.py`** — Bridges the model to the CUDA rasterizer. Projects 3D Gaussians to screen, evaluates SH colors, alpha-blends, and optionally applies depth regularization and exposure compensation.

- **`arguments/__init__.py`** — Three parameter groups: `ModelParams` (dataset paths, SH degree, resolution), `PipelineParams` (rasterizer options, antialiasing), `OptimizationParams` (learning rates, densification schedule, depth/exposure settings).

- **`scene/__init__.py`** — Scene management: loads cameras and point clouds, manages train/test splits, handles checkpoint save/load.

### Submodules
- **`submodules/diff-gaussian-rasterization`** — The CUDA differentiable rasterizer and core rendering engine. Projects 3D Gaussians onto the screen, evaluates spherical harmonics for view-dependent color, and performs alpha blending. Differentiable so gradients flow back through rendering for training. The default branch (`dr_aa`) supports anti-aliasing; the `3dgs_accel` branch adds a sparse Adam optimizer for ~2.7x training speedup.
- **`submodules/simple-knn`** — K-nearest-neighbors implementation used during initialization to compute distances between SfM points. These distances determine the initial scale of each Gaussian so they cover the scene appropriately.
- **`submodules/fused-ssim`** — Fused CUDA implementation of SSIM loss. Replaces the standard PyTorch SSIM computation with a faster kernel, reducing overhead since SSIM is evaluated every iteration as part of the loss.
- **`SIBR_viewers`** — C++/OpenGL viewer framework. Provides a **network viewer** (`SIBR_remoteGaussian_app`) that connects to a running training process for live visualization, and a **real-time viewer** (`SIBR_gaussianViewer_app`) that renders trained models at interactive framerates. Standalone C++ application built with CMake.

### Density Control
Gaussians are added/removed during training (iters 500–15000):
- **Clone**: small Gaussians with large view-space gradients are duplicated
- **Split**: large Gaussians with large gradients are split into smaller ones
- **Prune**: near-transparent Gaussians (opacity < threshold) are removed
- **Opacity reset**: every 3000 iterations, all opacities are reset to prevent floaters

### Switching to Accelerated Rasterizer
```bash
pip uninstall diff-gaussian-rasterization -y
cd submodules/diff-gaussian-rasterization && rm -rf build && git checkout 3dgs_accel && pip install . && cd ../..
```
Enables `--optimizer_type sparse_adam` for ~2.7x training speedup. The default `dr_aa` branch supports anti-aliasing.

## Key Parameters

- `--optimizer_type sparse_adam` — Requires the `3dgs_accel` rasterizer branch; significant speedup
- `--antialiasing` — EWA filter from Mip-Splatting
- `--data_device cpu` — Keep images on CPU to reduce VRAM usage
- `--resolution {1,2,4,8}` — Downsample factor; auto-downscales if width > 1600px
- `-d <depth_dir>` + `--depth_l1_weight_init` — Depth map regularization

## COLMAP Dataset Structure
```
<location>/
├── images/          # Input images
└── sparse/0/        # cameras.bin, images.bin, points3D.bin
```
Camera models must be SIMPLE_PINHOLE or PINHOLE. Use `convert.py` to create this from raw images.

## Requirements

- CUDA GPU with Compute Capability 7.0+, 24 GB VRAM recommended (reduce with higher `--densify_grad_threshold` or `--data_device cpu`)
- Python 3.7.13, PyTorch 1.12.1, CUDA SDK 11 (not 12); also works with Python 3.8 / PyTorch 2.0 / CUDA 12
- Datasets must be in COLMAP or Blender NeRF Synthetic format
