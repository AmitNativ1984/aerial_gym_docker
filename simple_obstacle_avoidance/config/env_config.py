"""
Environment configuration for simple obstacle avoidance.
Defines a sparse obstacle environment with only 5 obstacles.
"""
from aerial_gym.config.asset_config.env_object_config import (
    asset_state_params,
    left_wall,
    right_wall,
    back_wall,
    front_wall,
    bottom_wall,
    top_wall,
    tree_asset_params
)
from aerial_gym import AERIAL_GYM_DIRECTORY
import numpy as np


class sparse_object_params(asset_state_params):
    """
    Sparse obstacle configuration - only 5 obstacles (vs 35 in default env).
    Obstacles are spread across the middle of the environment.
    """
    num_assets = 5
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    keep_in_env = True  # Keep obstacles consistent across resets

    # Spread obstacles across middle of env (30%-70% of X range)
    # State vector: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, scale,
    #                lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
    min_state_ratio = [
        0.30, 0.10, 0.10,           # position x, y, z (as ratio of bounds)
        -np.pi, -np.pi, -np.pi,     # rotation (full range)
        1.0,                         # scale
        0.0, 0.0, 0.0,              # linear velocity (static)
        0.0, 0.0, 0.0,              # angular velocity (static)
    ]
    max_state_ratio = [
        0.70, 0.90, 0.90,           # position x, y, z
        np.pi, np.pi, np.pi,        # rotation
        1.0,                         # scale
        0.0, 0.0, 0.0,              # linear velocity
        0.0, 0.0, 0.0,              # angular velocity
    ]

    collision_mask = 1
    per_link_semantic = False
    semantic_id = -1


class SimpleObstacleEnvCfg:
    """
    Simple obstacle avoidance environment configuration.
    - 5 sparse obstacles in the middle of the environment
    - Boundary walls on all 6 sides
    - Reset on collision enabled
    """

    class env:
        num_envs = 64  # Overridden by task config
        num_env_actions = 4  # Action dimension
        env_spacing = 5.0  # Spacing between parallel environments

        # Physics simulation
        num_physics_steps_per_env_step_mean = 10
        num_physics_steps_per_env_step_std = 0

        # Rendering
        render_viewer_every_n_steps = 1

        # Collision handling
        reset_on_collision = True
        collision_force_threshold = 0.05  # Newtons

        # Environment setup
        create_ground_plane = False
        sample_timestep_for_latency = True
        perturb_observations = True
        keep_same_env_for_num_episodes = 10
        write_to_sim_at_every_timestep = False

        # Use warp for rendering
        use_warp = True

        # Environment bounds (same as env_with_obstacles)
        # These define the 3D space where the drone operates
        lower_bound_min = [-2.0, -4.0, -3.0]
        lower_bound_max = [-1.0, -2.5, -2.0]
        upper_bound_min = [9.0, 2.5, 2.0]
        upper_bound_max = [10.0, 4.0, 3.0]

    class env_config:
        # Enable/disable asset types
        include_asset_type = {
            # Only enable sparse objects
            "objects": True,
            # Disable other obstacle types
            "panels": False,
            "tiles": False,
            "thin": False,
            "trees": False,
            # Keep boundary walls
            "left_wall": False,
            "right_wall": False,
            "back_wall": False,
            "front_wall": False,
            "top_wall": False,
            "bottom_wall": True,
        }

        # Map asset type names to their configuration classes
        asset_type_to_dict_map = {
            "objects": sparse_object_params,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "back_wall": back_wall,
            "front_wall": front_wall,
            "bottom_wall": bottom_wall,
            "top_wall": top_wall,
            "trees": tree_asset_params
        }
