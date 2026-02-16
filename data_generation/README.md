# Depth Image Dataset Generation for VAE Training

Generates a dataset of 85,000 depth images for supervised training of a Variational Autoencoder (VAE), following the approach in [Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding](https://arxiv.org/abs/2402.03947).

## Overview

A simulated Intel RealSense D435 depth camera is randomly positioned inside environments filled with diverse obstacles. The aerial gym simulator (built on NVIDIA Isaac Gym) renders depth images using GPU-accelerated Warp ray-casting across multiple parallel environments.

Each capture randomizes:
- **Camera pose**: random position (10-90% of env bounds) and orientation (roll/pitch +/-60 deg, yaw full 360 deg)
- **Obstacle layout**: all obstacle positions and orientations re-randomized every step
- **Obstacle density**: number of active obstacles varies randomly per step (from sparse to dense)
- **Environment bounds**: volume dimensions randomized per-environment (~18-25m x 14-20m x 8-14m)

## Camera Parameters

Simulates the Intel RealSense D435 depth camera:

| Parameter       | Value       |
|-----------------|-------------|
| Resolution      | 1280 x 720  |
| Horizontal FOV  | 87 deg      |
| Vertical FOV    | ~58 deg (derived from aspect ratio) |
| Min range       | 0.105 m     |
| Max range       | 10.0 m      |

## Obstacle Types

| Type     | Count per env | Source URDFs | Description |
|----------|---------------|--------------|-------------|
| Panels   | 15            | 51 varied URDFs | Flat surfaces with random width (0.5-5.0m), height (1.0-6.0m), and thickness (0.05-0.15m). Full rotation randomization. |
| Thin     | 8             | 1000 URDFs   | Procedurally generated thin obstacles with varied shapes. |
| Trees    | 4             | 100 URDFs    | Procedurally generated tree structures made of cylinder branches. Always present in scene. |
| Objects  | 50            | 4 URDFs      | Small cubes, cuboidal rods, and wall segments (0.4-2.0m). Full rotation randomization. |

The asset loader randomly selects URDFs from each type's folder at simulation creation. On each reset, all non-tree obstacle positions and orientations are re-randomized, and a random subset may be culled (moved off-screen) to vary scene density.

## Output Format

- **16-bit grayscale PNG** (default): Normalized depth values scaled to 0-65535. To recover depth in meters: `depth_m = (pixel_value / 65535.0) * 10.0`.
- **NumPy float32** (optional): Raw normalized depth values in [0, 1]. Use `--format npy`.

Images with fewer than 5% valid pixels (camera inside obstacle or pointing into empty space) are automatically skipped.

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker container running with the aerial gym simulator (see main repo README)
- Panels must be generated first (see below)

## Usage

### 1. Generate varied panel URDFs (one-time setup)

```bash
python data_generation/generate_panels.py
```

This creates 50 panel URDFs with randomized dimensions in the aerial gym assets folder.

### 2. Generate the dataset

```bash
# Full dataset (85,000 images, default settings)
python -m data_generation.generate_dataset

# Custom settings
python -m data_generation.generate_dataset \
    --num_images 85000 \
    --num_envs 16 \
    --output_dir ~/DATA/depth-images \
    --format png \
    --seed 42

# Quick test run
python -m data_generation.generate_dataset --num_images 32 --num_envs 4
```

### CLI Arguments

| Argument            | Default                  | Description |
|---------------------|--------------------------|-------------|
| `--num_images`      | 85000                    | Total number of depth images to generate |
| `--num_envs`        | 16                       | Number of parallel environments. Reduce if out of memory. |
| `--output_dir`      | `~/DATA/depth-images`    | Output directory |
| `--format`          | `png`                    | `png` (16-bit) or `npy` (float32) |
| `--device`          | `cuda:0`                 | CUDA device |
| `--seed`            | 42                       | Random seed for reproducibility |
| `--min_valid_ratio` | 0.05                     | Min fraction of valid pixels to accept an image |

### VRAM Guidelines

| GPU VRAM | Recommended `--num_envs` |
|----------|--------------------------|
| 8 GB     | 4-8                      |
| 16 GB    | 16-32                    |
| 24 GB    | 32-64                    |

## File Structure

```
data_generation/
├── __init__.py              # urdfpy monkey-patch + registry registration
├── config/
│   ├── __init__.py
│   ├── camera_config.py     # Intel RealSense D435 parameters
│   ├── robot_config.py      # Camera carrier quadrotor with wide pose randomization
│   └── env_config.py        # Environment with all obstacle types, no fixed walls
├── generate_dataset.py      # Main data collection script
├── generate_panels.py       # One-time panel URDF generation
└── README.md
```

## Technical Notes

- **urdfpy monkey-patch**: The `__init__.py` patches a bug in `urdfpy` where `Cylinder`, `Box`, and `Sphere` geometry classes crash on `len(self._meshes)` when `_meshes` is `None`. This is required for loading tree URDFs (which use cylinder primitives).
- **Warp renderer**: Uses `use_warp=True` for GPU-accelerated ray-casting, which is significantly faster than Isaac Gym's native camera sensors for large numbers of environments.
- **No physics needed**: The quadrotor has gravity disabled and receives zero actions. A single physics step is run per capture only to update the simulation state after reset.
