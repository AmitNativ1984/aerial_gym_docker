import torch


def beta_schedule(epoch: int, warmup_epochs: int, beta_target: float) -> float:
    """Linear KL beta warmup: 0 -> beta_target over warmup_epochs."""
    if warmup_epochs <= 0:
        return beta_target
    return min(beta_target, beta_target * epoch / warmup_epochs)


def vae_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    obstacle_weight: float = 5.0,
    obstacle_threshold: float = 0.05,
    range_weight_power: float = 1.0,
):
    """VAE loss = weighted reconstruction + beta * KL divergence.

    Args:
        x_recon: [B, 1, H, W] reconstructed normalized depth.
        x_target: [B, 1, H, W] ground-truth normalized depth (post-dilation).
        mu: [B, latent_dim] latent means.
        logvar: [B, latent_dim] latent log-variances.
        beta: KL divergence weight.
        obstacle_weight: Extra weight for obstacle pixels (normalized > threshold).
        obstacle_threshold: Pixels above this value are considered obstacles.
        range_weight_power: Exponent for distance-proportional weighting.
            Since inverse depth normalization maps near=~1.0 and far=~0.0,
            weight = 1 + obstacle_weight * x_target^power for obstacle pixels.
            Higher power concentrates weight on the nearest obstacles.
            Set to 0.0 for the old binary weighting behavior.

    Returns:
        (total_loss, recon_loss, kl_loss) all scalar tensors.
    """
    # Distance-proportional weighting: near obstacles get highest weight.
    # x_target is in inverse depth space: near ~1.0, far ~0.0.
    obstacle_mask = x_target > obstacle_threshold
    weight = torch.ones_like(x_target)
    if range_weight_power > 0.0:
        # Continuous weight: 1 + obstacle_weight * value^power (near pixels get up to 1+obstacle_weight)
        weight[obstacle_mask] = 1.0 + obstacle_weight * x_target[obstacle_mask].pow(range_weight_power)
    else:
        # Legacy binary weighting
        weight[obstacle_mask] = obstacle_weight

    sq_error = (x_recon - x_target) ** 2
    recon_loss = (weight * sq_error).mean()

    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_per_dim.mean(dim=1).mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
