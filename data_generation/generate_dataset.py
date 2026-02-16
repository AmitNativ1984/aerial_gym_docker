"""Generate depth image dataset for VAE training.

Captures 85,000 depth images at 1280x720 (Intel RealSense D435 parameters)
from the aerial gym simulator with randomly positioned cameras and diverse obstacles.

Usage:
    python -m data_generation.generate_dataset
    python -m data_generation.generate_dataset --num_images 85000 --num_envs 16
    python -m data_generation.generate_dataset --num_images 32 --num_envs 4  # quick test
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("data_generation")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate depth image dataset for VAE training")
    parser.add_argument(
        "--num_images",
        type=int,
        default=85000,
        help="Total number of images to generate",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments (reduce if OOM)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/DATA/depth-images"),
        help="Output directory for depth images",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "npy"],
        help="Image save format: png (16-bit) or npy (float32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.05,
        help="Minimum ratio of valid pixels (0-1) to accept an image",
    )
    return parser.parse_args()


def setup_environment(args):
    """Build the simulation environment with custom data generation configs."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Import registers our custom configs with aerial_gym registries
    import data_generation  # noqa: F401

    from aerial_gym.sim.sim_builder import SimBuilder

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="data_gen_env",
        robot_name="data_gen_quad",
        controller_name="lee_velocity_control",
        args=None,
        device=args.device,
        num_envs=args.num_envs,
        headless=True,
        use_warp=True,
    )
    return env_manager


def save_depth_image(depth_tensor, filepath, fmt="png"):
    """Save a single depth image tensor to disk.

    Args:
        depth_tensor: [H, W] tensor with normalized depth values in [0, 1]
        filepath: output file path (without extension)
        fmt: 'png' (16-bit grayscale) or 'npy' (float32)
    """
    depth_np = depth_tensor.cpu().numpy()

    if fmt == "npy":
        np.save(filepath + ".npy", depth_np.astype(np.float32))
    else:
        # 16-bit PNG: clamp to [0,1], scale to [0, 65535]
        depth_clamped = np.clip(depth_np, 0.0, 1.0)
        depth_uint16 = (depth_clamped * 65535.0).astype(np.uint16)
        img = Image.fromarray(depth_uint16, mode="I;16")
        img.save(filepath + ".png")


def generate_dataset(args):
    """Main dataset generation loop."""
    os.makedirs(args.output_dir, exist_ok=True)

    logger.warning(f"Generating {args.num_images} depth images with {args.num_envs} parallel envs")
    logger.warning(f"Output directory: {args.output_dir}")
    logger.warning(f"Format: {args.format}, min valid ratio: {args.min_valid_ratio}")

    env_manager = setup_environment(args)

    # Zero actions -- robot is static, we only need to render
    actions = torch.zeros((args.num_envs, 4), device=args.device)

    # Track total obstacle count and keep_in_env count for density randomization.
    # keep_in_env assets (walls, panels, trees) are always present.
    # Non-keep_in_env assets (objects, thin) can be randomly culled.
    total_obstacles = env_manager.global_tensor_dict["num_obstacles_in_env"]
    keep_in_env_count = env_manager.asset_manager.num_keep_in_env

    # Initial reset
    env_manager.reset()

    total_saved = 0
    total_skipped = 0
    step_count = 0
    start_time = time.time()

    while total_saved < args.num_images:
        # Randomize obstacle density: between keep_in_env count and total obstacles
        random_num_obs = torch.randint(
            keep_in_env_count, total_obstacles + 1, (1,)
        ).item()
        env_manager.global_tensor_dict["num_obstacles_in_env"] = random_num_obs

        # Reset all environments for new random poses and obstacle layouts
        env_manager.reset()

        # Step physics (minimal, required to update state)
        env_manager.step(actions=actions)

        # Render depth cameras
        env_manager.render(render_components="sensors")

        # Reset any terminated/truncated envs
        env_manager.reset_terminated_and_truncated_envs()

        # Extract depth images: shape [num_envs, num_sensors, H, W]
        depth_tensor = env_manager.global_tensor_dict["depth_range_pixels"]

        # Save each environment's image
        for env_idx in range(args.num_envs):
            if total_saved >= args.num_images:
                break

            img_tensor = depth_tensor[env_idx, 0]  # [H, W], sensor 0

            # Validate: skip images where camera is inside an obstacle
            # Valid pixels have depth in (0, 1) -- not at min or max range
            valid_pixels = (img_tensor > 0) & (img_tensor < 1.0)
            valid_ratio = valid_pixels.float().mean().item()

            if valid_ratio < args.min_valid_ratio:
                total_skipped += 1
                continue

            filepath = os.path.join(args.output_dir, f"depth_{total_saved:06d}")
            save_depth_image(img_tensor, filepath, fmt=args.format)
            total_saved += 1

        step_count += 1

        # Progress logging every 50 steps
        if step_count % 50 == 0:
            elapsed = time.time() - start_time
            rate = total_saved / elapsed if elapsed > 0 else 0
            logger.warning(
                f"Step {step_count}: {total_saved}/{args.num_images} images saved, "
                f"{total_skipped} skipped ({rate:.1f} img/s)"
            )

    elapsed = time.time() - start_time
    logger.warning(
        f"Dataset generation complete: {total_saved} images, "
        f"{total_skipped} skipped, {elapsed:.1f}s total, "
        f"{total_saved / elapsed:.1f} img/s"
    )


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
