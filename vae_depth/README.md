# Depth VAE with Deep Collision Encoding (DCE)

Variational Autoencoder that compresses 320x180 depth images into a 32-dimensional latent vector for drone navigation RL. The key idea: the encoder receives **raw depth** while the decoder learns to reconstruct **collision-dilated depth**, so the latent space implicitly encodes collision safety without requiring dilation at inference time.

## Method

### Deep Collision Encoding

Standard depth VAEs encode and reconstruct the same image. DCE changes this: the encoder input is the original depth image, but the reconstruction target is a min-pool dilated version that expands obstacles by the drone's collision radius. This forces the latent representation to internalize obstacle safety margins.

At RL inference time, the encoder receives raw depth from the simulator and outputs a latent vector that already accounts for the drone's physical size -- no dilation preprocessing needed.

### Preprocessing Pipeline

**Encoder input** (raw depth):
1. Load 16-bit PNG depth image (Intel RealSense D435, 1280x720)
2. Convert to meters: `pixel_value / 65535 * depth_scale`
3. Resize to 320x180 (nearest-neighbor)
4. Linear normalization: `normalized = 1 - depth / max_depth` (near=1, far=0)

**Decoder target** (dilated depth):
1. Same as above, but after step 3:
2. Min-pool dilation with kernel derived from drone geometry (expands obstacles)
3. Then normalize

The dilation kernel size is auto-computed from:
- Drone radius: 0.25m (half widest dimension including propellers)
- Safety margin: 50% of drone radius
- Reference distance: 3m
- Camera HFOV: 87 degrees

### Loss Function

```
loss = weighted_MSE(predicted, dilated_target) + beta * KL_divergence
```

- **Weighted MSE**: Distance-proportional weighting gives higher weight to near obstacles. Weight = `1 + obstacle_weight * normalized_depth^power` for all pixels.
- **KL divergence**: Standard VAE regularization with beta warmup (0 -> 0.001 over 10 epochs) to prevent posterior collapse.

### Architecture

**Encoder**: 4 convolutional blocks (32->64->128->256 channels), each with two 3x3 convs + BatchNorm + ELU. Followed by 1x1 channel reduction, flatten, and FC head (512 -> 2*latent_dim). Outputs concatenated mu and logvar.

**Decoder**: FC head (latent_dim -> 512 -> reshape 6x10), 1x1 channel expansion, then 5 transposed conv layers (256->128->128->64->32->1) with BatchNorm + ELU. Final Sigmoid activation.

**Latent space**: 32 dimensions. During training, uses reparameterization trick for sampling. During RL inference, uses mu only (deterministic).

## Dataset Generation

Depth images are generated using the Aerial Gym Simulator (Isaac Gym):

```bash
python -m data_generation.generate_dataset --num_images 85000 --num_envs 16
```

This creates randomized environments with panels, thin structures, trees, and objects, then captures depth images from random drone poses. Output: 16-bit PNG files in `~/DATA/depth-images/`.

Camera specs match Intel RealSense D435: 1280x720, 87 degrees HFOV, 0.1-10m range.

## Training

```bash
# Smoke test
python -m vae_depth.train --num_epochs 2 --batch_size 16

# Full training
python -m vae_depth.train --num_epochs 100 --batch_size 64

# Monitor
tensorboard --logdir vae_depth/runs/
```

TensorBoard shows 3-column reconstruction images (Raw Depth | Dilated GT | Predicted), error heatmaps, per-dimension KL divergence, and active latent dimension count.

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| latent_dim | 32 | Latent space dimensionality |
| batch_size | 64 | Training batch size |
| learning_rate | 1e-4 | Adam optimizer LR |
| beta_target | 0.001 | KL divergence weight |
| obstacle_weight | 5.0 | Near-obstacle loss emphasis |
| max_depth_m | 7.0 | Depth clipping distance |

## Evaluation

```bash
python -m vae_depth.evaluate --checkpoint vae_depth/runs/<timestamp>/checkpoints/best.pth
```

Generates reconstruction comparison images, error heatmaps, and latent space statistics.

## RL Integration

```python
from vae_depth.vae_image_encoder import DepthVAEImageEncoder

encoder = DepthVAEImageEncoder(config, device="cuda:0")
latents = encoder.encode(depth_images)  # [num_envs, 32]
```

The encoder receives raw depth from the simulator (no dilation needed) and outputs a 32-dim latent vector with implicit collision safety.
