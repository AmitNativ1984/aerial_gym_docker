"""RL inference wrapper for the trained Depth VAE.

Drop-in replacement for the existing VAEImageEncoder at
/app/aerial_gym/aerial_gym_simulator/aerial_gym/utils/vae/vae_image_encoder.py

Usage during RL:
    encoder = DepthVAEImageEncoder(config, device="cuda:0")
    latents = encoder.encode(depth_images)  # [num_envs, 32]
"""

import torch
import torch.nn.functional as F

from vae_depth.model import DepthVAE
from vae_depth.preprocessing import min_pool_dilation, normalize_depth


class DepthVAEImageEncoder:
    """Wraps the trained DepthVAE encoder for frozen inference during RL.

    Pipeline:
        1. Receive [num_envs, H, W] depth from simulator (normalized [0,1] by sensor)
        2. Scale to meters
        3. Resize to target resolution (nearest)
        4. Min-pool dilation
        5. Inverse depth normalization
        6. Encoder forward -> [num_envs, latent_dim] (mu only, deterministic)
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = device

        # Load full VAE, extract encoder
        vae = DepthVAE(latent_dim=config.latent_dims)
        checkpoint = torch.load(config.model_file, map_location=device)
        vae.load_state_dict(checkpoint["model_state_dict"])

        self.encoder = vae.encoder.to(device)
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.latent_dim = config.latent_dims
        self.target_res = (config.target_height, config.target_width)
        self.dilation_kernel_size = config.dilation_kernel_size
        self.max_depth_m = config.max_depth_m
        self.min_depth_m = config.min_depth_m
        self.sensor_max_range = getattr(config, "sensor_max_range", 10.0)

    def encode(self, image_tensors):
        """Encode depth images to latent vectors.

        Args:
            image_tensors: [num_envs, H, W] or [num_envs, 1, H, W]
                Raw depth from simulator (values = depth_m / sensor_max_range).

        Returns:
            [num_envs, latent_dim] latent vectors (mu, deterministic).
        """
        with torch.no_grad():
            x = image_tensors
            if x.dim() == 3:
                x = x.unsqueeze(1)  # [B, 1, H, W]

            # Simulator normalized -> meters
            x = x * self.sensor_max_range

            # Resize to target resolution
            if x.shape[-2:] != self.target_res:
                x = F.interpolate(x, self.target_res, mode="nearest")

            # Min-pool dilation
            x = min_pool_dilation(x, self.dilation_kernel_size)

            # Inverse depth normalization
            x = normalize_depth(x, self.max_depth_m, self.min_depth_m)

            # Encode -> [B, 2*latent_dim], take mu (first half)
            z_params = self.encoder(x)
            mu = z_params[:, : self.latent_dim]

            return mu

    def get_latent_dims_size(self):
        return self.latent_dim
