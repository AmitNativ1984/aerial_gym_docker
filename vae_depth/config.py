import os
import math
from dataclasses import dataclass, field


@dataclass
class VAEConfig:
    # Data
    data_dir: str = os.path.expanduser("~/DATA/depth-images/")
    image_format: str = "png"
    source_height: int = 720
    source_width: int = 1280

    target_height: int = 180
    target_width: int = 320

    # Depth normalization
    max_depth_m: float = 7.0
    min_depth_m: float = 0.1
    depth_scale: float = 10.0  # pixel_value / 65535 * depth_scale = meters

    # Min-pool dilation (derived from drone geometry)
    drone_radius_m: float = 0.25
    safety_margin_fraction: float = 0.5
    reference_distance_m: float = 3.0
    hfov_deg: float = 87.0
    dilation_kernel_size: int = 0  # 0 = auto-compute from drone params

    # Augmentation
    crop_prob: float = 0.5
    crop_scale_min: float = 0.8  # random crop keeps 80-100% of the resized image
    crop_scale_max: float = 1.0
    flip_prob: float = 0.5

    # VAE architecture
    latent_dim: int = 32
    encoder_channels: list = field(default_factory=lambda: [32, 64, 128, 256, 256])
    fc_hidden_dim: int = 256

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    val_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True

    # Loss
    beta_target: float = 0.001
    beta_warmup_epochs: int = 10
    obstacle_weight: float = 5.0
    obstacle_threshold: float = 0.0  # 0 = fully continuous weighting across all depths
    range_weight_power: float = 1.0  # exponent for distance-proportional weighting (0 = binary)

    # LR schedule
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-5
    lr_threshold: float = 5e-4

    # Checkpointing
    checkpoint_dir: str = "vae_depth/checkpoints"
    checkpoint_interval: int = 10

    # TensorBoard
    log_dir: str = "vae_depth/runs"
    image_log_interval: int = 5  # log reconstruction images every N epochs

    # Device
    device: str = "cuda:0"
    seed: int = 42

    def __post_init__(self):
        if self.dilation_kernel_size == 0:
            self.dilation_kernel_size = compute_dilation_kernel(
                drone_radius_m=self.drone_radius_m,
                safety_margin_fraction=self.safety_margin_fraction,
                reference_distance_m=self.reference_distance_m,
                hfov_deg=self.hfov_deg,
                image_width=self.target_width,
            )


def compute_dilation_kernel(
    drone_radius_m: float,
    safety_margin_fraction: float,
    reference_distance_m: float,
    hfov_deg: float,
    image_width: int,
) -> int:
    """Compute min-pool dilation kernel size from drone geometry.

    Args:
        drone_radius_m: Half the widest drone dimension including propellers.
        safety_margin_fraction: Fraction of drone radius to use as dilation margin.
        reference_distance_m: Distance at which the kernel size is calibrated.
        hfov_deg: Camera horizontal field of view in degrees.
        image_width: Image width in pixels.

    Returns:
        Odd integer kernel size (minimum 3).
    """
    hfov_rad = math.radians(hfov_deg)
    pixel_size_m = 2.0 * reference_distance_m * math.tan(hfov_rad / 2.0) / image_width
    margin_m = safety_margin_fraction * drone_radius_m
    margin_pixels = math.ceil(margin_m / pixel_size_m)
    kernel_size = 2 * margin_pixels + 1
    return max(kernel_size, 3)
