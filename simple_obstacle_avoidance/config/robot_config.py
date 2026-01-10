"""
Custom robot configuration with modified spawn position.
Inherits from BaseQuadWithCameraCfg and only overrides spawn location.
"""
import numpy as np
from aerial_gym.config.robot_config.base_quad_config import BaseQuadWithCameraCfg


class CustomQuadWithCameraCfg(BaseQuadWithCameraCfg):
    """
    Quadrotor with camera, but with custom spawn position.
    Only overrides the init_config class to change where drone spawns.
    """

    class init_config(BaseQuadWithCameraCfg.init_config):
        # Spawn position as ratio of environment bounds [x, y, z]
        # Original: x=0.1-0.3, y=0.1-0.9, z=0.1-0.9
        #
        # Change these values to move the spawn zone:
        #   0.0 = minimum bound (e.g., X=-2m)
        #   1.0 = maximum bound (e.g., X=+10m)
        min_state_ratio = [
            0.05,  # X: 5% (spawn near back wall)
            0.3,   # Y: 30%
            0.3,   # Z: 30%
            0,     # Roll (rad)
            0,     # Pitch (rad)
            -np.pi,  # Yaw min (rad) - facing any direction
            1.0,   # Scale (unused, keep at 1.0)
            0,     # Linear velocity X
            0,     # Linear velocity Y
            0,     # Linear velocity Z
            0,     # Angular velocity X
            0,     # Angular velocity Y
            0,     # Angular velocity Z
        ]
        max_state_ratio = [
            0.15,  # X: 15% (spawn zone width)
            0.7,   # Y: 70%
            0.7,   # Z: 70%
            0,     # Roll (rad)
            0,     # Pitch (rad)
            np.pi,   # Yaw max (rad)
            1.0,   # Scale
            0,     # Linear velocity X
            0,     # Linear velocity Y
            0,     # Linear velocity Z
            0,     # Angular velocity X
            0,     # Angular velocity Y
            0,     # Angular velocity Z
        ]
