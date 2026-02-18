import torch
import torch.nn.functional as F


def normalize_depth(depth_m: torch.Tensor, max_depth: float = 7.0, min_depth: float = 0.1) -> torch.Tensor:
    """Linear depth normalization.

    Near objects map to ~1.0, far objects to ~0.0, beyond max_depth clips to 0.0.
    Uniform precision across the full depth range.

    Args:
        depth_m: Depth in meters, any shape.
        max_depth: Clip depth at this distance (meters). Beyond this -> 0.
        min_depth: Floor depth (meters). Below this -> 1.0.

    Returns:
        Normalized tensor in [0, 1].
    """
    depth_clamped = torch.clamp(depth_m, min=min_depth, max=max_depth)
    normalized = 1.0 - depth_clamped / max_depth
    return normalized


def denormalize_depth(normalized: torch.Tensor, max_depth: float = 7.0, min_depth: float = 0.1) -> torch.Tensor:
    """Inverse of normalize_depth. Recovers metric depth from normalized values.

    Args:
        normalized: Values in [0, 1].
        max_depth: Same max_depth used during normalization.
        min_depth: Same min_depth used during normalization.

    Returns:
        Depth in meters.
    """
    depth_m = (1.0 - normalized) * max_depth
    return torch.clamp(depth_m, min=min_depth)


def min_pool_dilation(depth: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply min-pooling dilation to expand obstacles in a depth image.

    Replaces each pixel with the minimum depth in its neighborhood,
    effectively growing obstacle regions by (kernel_size-1)/2 pixels.

    Args:
        depth: Depth tensor of shape [B, 1, H, W] in meters (lower = closer = obstacle).
        kernel_size: Odd integer kernel size for the pooling window.

    Returns:
        Dilated depth tensor, same shape as input.
    """
    padding = kernel_size // 2
    # min-pool = -max_pool(-x)
    dilated = -F.max_pool2d(-depth, kernel_size=kernel_size, stride=1, padding=padding)
    return dilated
