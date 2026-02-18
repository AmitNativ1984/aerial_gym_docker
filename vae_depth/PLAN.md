# VAE Depth Image Encoder — Implementation Plan

## Context

Train a Variational Autoencoder (VAE) to compress 320x180 depth images into a 32-dim latent vector for downstream autonomous drone navigation. The existing codebase has a ResNet8-based VAE (64-dim latent, 270x480 input) at `/app/aerial_gym/aerial_gym_simulator/aerial_gym/utils/vae/VAE.py`. We're building a new, purpose-built VAE with: smaller input, half the latent dims, min-pooling dilation for safety margins, and inverse depth normalization for near-range precision.

**Dataset**: 43,088 16-bit PNG depth images at 1280x720 in `~/DATA/depth-images/`

## Project Structure

```
/workspaces/aerial_gym_docker/vae_depth/
├── __init__.py
├── config.py              # Dataclass with all hyperparameters
├── preprocessing.py       # normalize_depth, denormalize_depth, min_pool_dilation
├── dataset.py             # PyTorch Dataset (load PNGs, augment, preprocess)
├── model.py               # DepthEncoder, DepthDecoder, DepthVAE
├── loss.py                # Weighted MSE + KL with beta schedule
├── train.py               # Main training entry point
├── evaluate.py            # Reconstruction visualization & latent analysis
├── vae_image_encoder.py   # Inference wrapper for RL (drop-in replacement)
├── runs/                  # TensorBoard log directory (auto-created)
├── checkpoints/           # Model checkpoints (auto-created)
└── PLAN.md                # This plan document
```

---

## Step 1: `config.py` — Configuration Dataclass

Single source of truth for all parameters. CLI args override defaults.

Key values:
- `target_height=180, target_width=320`
- `latent_dim=32`
- `max_depth_m=7.0, min_depth_m=0.1`
- **Dilation params** (derived from drone geometry):
  - `drone_radius_m=0.25` — half the widest dimension including propellers
  - `safety_margin_fraction=0.5` — what fraction of the drone radius to dilate (tune this)
  - `reference_distance_m=3.0` — distance at which the kernel size is "correct"
  - `hfov_deg=87.0` — camera horizontal FOV (D435)
  - `dilation_kernel_size` — **computed** from the above: `2 * ceil(safety_margin_fraction * drone_radius_m / (reference_distance_m * 2 * tan(hfov/2) / target_width)) + 1`
  - With defaults (0.25m radius, 50% margin, 3m ref): effective margin = 0.125m ≈ 7 pixels → **kernel_size = 15**
  - Can be overridden directly via CLI `--dilation_kernel_size`
- `crop_prob=0.7` (70% random crop, 30% full resize)
- `batch_size=64, lr=1e-4, num_epochs=100`
- `beta_target=0.5, beta_warmup_epochs=10`
- `checkpoint_interval=10`
- `obstacle_weight=5.0` (for weighted MSE)

## Step 2: `preprocessing.py` — Depth Processing Functions

**Inverse depth normalization** (clipped at 7m):
```
normalized = (1/depth - 1/max_depth) / (1/min_depth - 1/max_depth)
```
- Near objects (0.1m) → ~1.0, far objects (7m) → 0.0, beyond 7m → clipped to 0.0
- ~10x more precision at 1m vs 5m
- `denormalize_depth()` provides the exact inverse for visualization

**Min-pool dilation** (safety margin expansion):
```
dilated = -F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size//2)
```
- Applied on metric depth BEFORE normalization
- Expands obstacles by `(kernel_size-1)/2` pixels in all directions

**Kernel size derivation from drone geometry:**
```
pixel_size_at_d = 2 * d * tan(hfov/2) / image_width     # meters per pixel at distance d
margin_m = safety_margin_fraction * drone_radius_m        # effective dilation in meters
margin_pixels = ceil(margin_m / pixel_size_at_d)          # pixels to dilate at reference distance
kernel_size = 2 * margin_pixels + 1                       # must be odd
```
With defaults (radius=0.25m, margin=50%, ref_dist=3m, HFOV=87°, width=320):
- pixel_size at 3m = 0.0178m/pixel
- margin = 0.125m → 7 pixels → kernel = 15

The `compute_dilation_kernel()` helper in `preprocessing.py` computes this from config.

All functions are pure PyTorch (GPU-compatible, shared between training and RL inference).

## Step 3: `dataset.py` — Data Loading & Augmentation

Pipeline per sample:
1. Load 16-bit PNG → float32 depth in meters (`pixel / 65535 * 10`)
2. **Augment** (training only):
   - 70% chance: random crop 640x360 from 1280x720 (2x zoom, simulates closer obstacles)
   - 30% chance: use full image
   - 50% chance: horizontal flip
3. Resize to 320x180 (nearest-neighbor, preserves depth edges)
4. Min-pool dilation (kernel computed from drone radius + safety margin config)
5. Inverse depth normalization → `[1, 180, 320]` tensor in [0, 1)

Returns `(processed_image, processed_image)` — VAE reconstructs the post-processed image.

Split: 90% train (~38,779), 10% val (~4,309), seeded random.

## Step 4: `model.py` — VAE Architecture (from scratch)

### Encoder: `[B, 1, 180, 320]` → `[B, 64]` (mu + logvar)

| Layer | Config | Output |
|-------|--------|--------|
| E1 | Conv2d(1→32, k=5, s=2, p=2) + BN + ELU | [B, 32, 90, 160] |
| E2 | Conv2d(32→64, k=3, s=2, p=1) + BN + ELU | [B, 64, 45, 80] |
| E3 | Conv2d(64→128, k=3, s=2, p=1) + BN + ELU | [B, 128, 23, 40] |
| E4 | Conv2d(128→256, k=3, s=2, p=1) + BN + ELU | [B, 256, 12, 20] |
| E5 | Conv2d(256→256, k=3, s=2, p=1) + BN + ELU | [B, 256, 6, 10] |
| Flatten | — | [B, 15360] |
| FC1 | Linear(15360→256) + ELU | [B, 256] |
| FC2 | Linear(256→64) | [B, 64] → split into mu[32], logvar[32] |

### Decoder: `[B, 32]` → `[B, 1, 180, 320]`

| Layer | Config | Output |
|-------|--------|--------|
| FC1 | Linear(32→256) + ELU | [B, 256] |
| FC2 | Linear(256→15360) + ELU | [B, 15360] |
| Reshape | — | [B, 256, 6, 10] |
| D5 | ConvT(256→256, k=3, s=2, p=1, op=(1,1)) + BN + ELU | [B, 256, 12, 20] |
| D4 | ConvT(256→128, k=3, s=2, p=1, op=(0,1)) + BN + ELU | [B, 128, 23, 40] |
| D3 | ConvT(128→64, k=3, s=2, p=1, op=(0,1)) + BN + ELU | [B, 64, 45, 80] |
| D2 | ConvT(64→32, k=3, s=2, p=1, op=(1,1)) + BN + ELU | [B, 32, 90, 160] |
| D1 | ConvT(32→1, k=5, s=2, p=2, op=(1,1)) + Sigmoid | [B, 1, 180, 320] |

Dimensions verified mathematically — encoder and decoder are exact mirrors.

**Interfaces**:
- `forward(x)` → `(x_recon, mu, logvar, z)` — training
- `encode(x)` → `mu` — deterministic inference (for RL)
- `decode(z)` → `x_recon` — visualization

## Step 5: `loss.py` — Weighted MSE + KL

```python
loss = weighted_recon_loss + beta * kl_loss
```

- **Weighted MSE**: 5x weight on obstacle pixels (normalized > 0.05), 1x on free space. Prevents the network from just learning to output zeros (89.5% of pixels are free space after clipping).
- **KL divergence**: Standard `−0.5 * Σ(1 + logvar − μ² − exp(logvar))`
- **Beta warmup**: Linear 0 → 0.5 over first 10 epochs (prevents KL collapse)

## Step 6: `train.py` — Training Loop

Entry point: `python -m vae_depth.train [--args]`

- Optimizer: Adam(lr=1e-4)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5, min_lr=1e-6)
- Gradient clipping: max_norm=1.0
- Logging: epoch loss, recon loss, KL loss, beta, LR to stdout + **TensorBoard**
- **TensorBoard logging** (to `vae_depth/runs/<timestamp>/`):
  - Scalar: total loss, recon loss, KL loss, beta, learning rate (per epoch, train & val)
  - Images: sample reconstructions vs ground truth (every N epochs, e.g., 5)
  - Histograms: latent mu and logvar distributions (per epoch)
  - Per-dimension KL values
- **Checkpoints**: Save `best.pth` (lowest val loss) + `epoch_N.pth` every 10 epochs
- Checkpoint contains: `model_state_dict`, `encoder_state_dict` (for easy RL extraction), `optimizer_state_dict`, `config`

## Step 7: `evaluate.py` — Visualization & Analysis

- Side-by-side input vs reconstruction images (saved as PNGs)
- Reconstruction error heatmaps
- Latent space statistics (per-dimension KL, mu/logvar distributions)
- Detects KL collapse (unused latent dimensions)

## Step 8: `vae_image_encoder.py` — RL Inference Wrapper

Drop-in replacement for existing `VAEImageEncoder`. Same `encode(image_tensors)` interface.

Pipeline at inference:
1. Receive `[num_envs, H, W]` depth from simulator (normalized [0, 1] by sensor)
2. Scale to meters (`× sensor_max_range`)
3. Resize to 320×180 (nearest)
4. Min-pool dilation
5. Inverse depth normalization
6. Encoder forward → `[num_envs, 32]` latent (mu only, deterministic)

Model is frozen (eval mode, no gradients).

---

## Implementation Order

1. Create `vae_depth/` directory + `PLAN.md` (this plan) + `__init__.py`
2. `config.py`
3. `preprocessing.py` + quick test
4. `model.py` + smoke test (random input, verify shapes)
5. `loss.py`
6. `dataset.py` + verify loading a few images
7. `train.py` (with TensorBoard integration) + run 2-epoch smoke test
8. `evaluate.py`
9. `vae_image_encoder.py`

## Verification

```bash
# Smoke test (inside Docker container)
python -m vae_depth.train --num_epochs 2 --batch_size 16

# Full training
python -m vae_depth.train --num_epochs 100 --batch_size 64

# Evaluate
python -m vae_depth.evaluate --checkpoint vae_depth/checkpoints/best.pth
```

**Success criteria**:
- Val reconstruction loss decreasing and stable
- All 32 latent dimensions active (per-dim KL > 0.1 nats)
- Reconstructions preserve obstacle structure (edges, relative distances)
- Encoder inference <1ms for batch of 4096 on GPU
