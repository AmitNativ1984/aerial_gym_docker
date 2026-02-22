"""Evaluate a trained Depth VAE checkpoint.

Usage:
    python -m vae_depth.evaluate --checkpoint vae_depth/checkpoints/best.pth

Generates:
    - Side-by-side input vs reconstruction images
    - Error heatmaps
    - Latent space statistics (per-dimension KL, active dims)
"""

import argparse
import os
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from vae_depth.config import VAEConfig
from vae_depth.dataset import DepthImageDataset
from vae_depth.model import DepthVAE
from vae_depth.preprocessing import denormalize_depth, min_pool_dilation, normalize_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Depth VAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--output_dir", type=str, default="vae_depth/eval_output", help="Output dir for images")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of reconstruction samples")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_config = ckpt["config"]
    config = VAEConfig()
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.__post_init__()

    model = DepthVAE(latent_dim=config.latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"Loaded checkpoint from epoch {epoch}, val_loss={val_loss}")
    return model, config


def plot_reconstructions(model, val_loader, device, config, output_dir, num_samples=16):
    """Save Original | Dilated GT | Predicted images side by side with colorbar."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        batch_depth_m = next(iter(val_loader))
        batch_depth_m = batch_depth_m[:num_samples].to(device)
        batch_input = normalize_depth(batch_depth_m, config.max_depth_m, config.min_depth_m)
        batch_target = min_pool_dilation(batch_depth_m, config.dilation_kernel_size)
        batch_target = normalize_depth(batch_target, config.max_depth_m, config.min_depth_m)
        x_recon, mu, logvar, z = model(batch_input)

    batch_input = batch_input.cpu()
    batch_target = batch_target.cpu()
    x_recon = x_recon.cpu()

    n = min(num_samples, 8)

    # --- Reconstructions: each row is [Input | Dilated GT | Predicted] ---
    fig, axes = plt.subplots(n, 3, figsize=(16, 2.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img_input = batch_input[i, 0].numpy()
        img_target = batch_target[i, 0].numpy()
        img_rec = x_recon[i, 0].numpy()

        axes[i, 0].imshow(img_input, cmap="turbo", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Raw Depth {i}", fontsize=9)
        axes[i, 0].axis("off")

        im_gt = axes[i, 1].imshow(img_target, cmap="turbo", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Dilated GT {i}", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_rec, cmap="turbo", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Predicted {i}", fontsize=9)
        axes[i, 2].axis("off")

    # Shared colorbar on the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im_gt, cax=cbar_ax)
    cbar.set_label("Normalized depth (near=1, far=0)", fontsize=8)

    path = os.path.join(output_dir, "reconstructions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstructions to {path}")

    # --- Error heatmaps: each row is [Input | Target | Predicted | Error] ---
    fig, axes = plt.subplots(n, 4, figsize=(20, 2.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img_input = batch_input[i, 0].numpy()
        img_target = batch_target[i, 0].numpy()
        img_rec = x_recon[i, 0].numpy()
        error = np.abs(img_target - img_rec)

        axes[i, 0].imshow(img_input, cmap="turbo", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Raw Depth {i}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_target, cmap="turbo", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Dilated GT {i}", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_rec, cmap="turbo", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Predicted {i}", fontsize=9)
        axes[i, 2].axis("off")

        im_err = axes[i, 3].imshow(error, cmap="hot", vmin=0, vmax=0.5)
        axes[i, 3].set_title(f"Error (MAE={error.mean():.4f})", fontsize=9)
        axes[i, 3].axis("off")

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im_err, cax=cbar_ax)
    cbar.set_label("Absolute error", fontsize=8)

    path = os.path.join(output_dir, "error_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error heatmaps to {path}")


def plot_latent_statistics(model, val_loader, device, config, output_dir):
    """Analyze latent space: per-dimension KL, mu/logvar distributions."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for batch_depth_m in val_loader:
            batch_depth_m = batch_depth_m.to(device)
            batch_input = normalize_depth(batch_depth_m, config.max_depth_m, config.min_depth_m)
            _, mu, logvar, _ = model(batch_input)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    # Per-dimension KL
    kl_per_dim = -0.5 * (1.0 + all_logvar - all_mu.pow(2) - all_logvar.exp())
    kl_mean = kl_per_dim.mean(dim=0).numpy()
    active_dims = (kl_mean > 0.1).sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # KL per dimension
    axes[0].bar(range(config.latent_dim), kl_mean)
    axes[0].axhline(y=0.1, color="r", linestyle="--", label="active threshold (0.1)")
    axes[0].set_xlabel("Latent dimension")
    axes[0].set_ylabel("Mean KL divergence")
    axes[0].set_title(f"Per-dim KL (active: {active_dims}/{config.latent_dim})")
    axes[0].legend()

    # Mu distribution
    mu_mean = all_mu.mean(dim=0).numpy()
    mu_std = all_mu.std(dim=0).numpy()
    axes[1].errorbar(range(config.latent_dim), mu_mean, yerr=mu_std, fmt="o", markersize=3)
    axes[1].axhline(y=0, color="gray", linestyle="--")
    axes[1].set_xlabel("Latent dimension")
    axes[1].set_ylabel("mu")
    axes[1].set_title("Latent mu (mean +/- std)")

    # Logvar distribution
    logvar_mean = all_logvar.mean(dim=0).numpy()
    logvar_std = all_logvar.std(dim=0).numpy()
    axes[2].errorbar(range(config.latent_dim), logvar_mean, yerr=logvar_std, fmt="o", markersize=3)
    axes[2].axhline(y=0, color="gray", linestyle="--")
    axes[2].set_xlabel("Latent dimension")
    axes[2].set_ylabel("logvar")
    axes[2].set_title("Latent logvar (mean +/- std)")

    plt.tight_layout()
    path = os.path.join(output_dir, "latent_statistics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved latent statistics to {path}")
    print(f"Active latent dimensions: {active_dims}/{config.latent_dim}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, config = load_model(args.checkpoint, device)
    if args.data_dir:
        config.data_dir = args.data_dir

    # Load validation set
    all_paths = sorted(glob(os.path.join(config.data_dir, f"*.{config.image_format}")))
    split_idx = int(len(all_paths) * (1 - config.val_split))
    rng = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(len(all_paths), generator=rng).tolist()
    val_paths = [all_paths[i] for i in perm[split_idx:]]
    print(f"Val set: {len(val_paths)} images")

    val_dataset = DepthImageDataset(val_paths, config, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    output_dir = args.output_dir
    plot_reconstructions(model, val_loader, device, config, output_dir, args.num_samples)
    plot_latent_statistics(model, val_loader, device, config, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
