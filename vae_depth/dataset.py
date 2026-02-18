import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DepthImageDataset(Dataset):
    """PyTorch Dataset for 16-bit PNG depth images.

    Loads images at source resolution, resizes to target resolution,
    then applies augmentation (random crop + resize back, horizontal flip).
    Returns raw depth in meters as [1, H, W] tensor.

    Min-pool dilation and normalization are applied on GPU in the training loop.
    """

    def __init__(self, image_paths: list, config, augment: bool = True):
        self.image_paths = image_paths
        self.config = config
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load 16-bit PNG -> float32 depth in meters (cv2 is ~2x faster than PIL)
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        depth_np = img.astype(np.float32) / 65535.0 * self.config.depth_scale

        # 2. Resize full image to target resolution (nearest-neighbor preserves depth edges)
        depth_np = cv2.resize(
            depth_np,
            (self.config.target_width, self.config.target_height),
            interpolation=cv2.INTER_NEAREST,
        )

        # 3. Augmentation: random crop a large portion, then resize back to target
        if self.augment and random.random() < self.config.crop_prob:
            scale = random.uniform(self.config.crop_scale_min, self.config.crop_scale_max)
            crop_h = int(self.config.target_height * scale)
            crop_w = int(self.config.target_width * scale)
            top = random.randint(0, self.config.target_height - crop_h)
            left = random.randint(0, self.config.target_width - crop_w)
            depth_np = depth_np[top : top + crop_h, left : left + crop_w]
            depth_np = cv2.resize(
                depth_np,
                (self.config.target_width, self.config.target_height),
                interpolation=cv2.INTER_NEAREST,
            )

        # 4. Optional horizontal flip
        if self.augment and random.random() < self.config.flip_prob:
            depth_np = np.flip(depth_np, axis=1).copy()

        # 5. Convert to tensor [1, H, W]
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        return depth_tensor
