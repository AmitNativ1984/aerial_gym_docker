"""
Environment configuration for simple hover SNN task.
Defines a ground plane, no obstacles.

Identical to simple_hover environment - the SNN is only
a different neural network architecture, not a different environment.
"""

from aerial_gym.config.asset_config.env_object_config import bottom_wall
from aerial_gym import AERIAL_GYM_DIRECTORY
import numpy as np

class SimpleHoverSNNEnvCfg:
    """
    Simple hover SNN environment configuration.
    - Ground plane only, no obstacles
    - Environment boundaries
    - Reset on crash or out-of-bounds enabled
    """

    class env:
        num_envs = 64  # Overridden by task config
        num_env_actions = 0     # this is the number of actions handled by the environment
                                # these are the actions that are sent to environment entities
                                # and some of them may be used to control various entities in the environment
                                # e.g. motion of obstacles, etc.
        env_spacing = 5.0 # 5.0 x 5.0 x 5.0 m spacing between envs

        # Environment setup
        create_ground_plane = True          # Create ground plane
        reset_out_of_bounds = True

        # Physics simulation
        num_physics_steps_per_env_step_mean = 5     # number of steps between camera renders mean
        num_physics_steps_per_env_step_std = 0      # number of steps between camera renders std

        # Rendering
        render_viewer_every_n_steps = 1
        use_warp = False     # Rendering for camera. Use warp if going to use camera images for RL

        manual_camera_trigger = False # if true, camera rendering is triggered manually from task

        # Collision handling
        reset_on_collision = True
        collision_force_threshold = 0.05  # Newtons

        sample_timestep_for_latency = True
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False

        # Environment bounds
        e_s = env_spacing
        lower_bound_min = [-e_s, -e_s, 0.0]  # X, Y, Z min (Z=0 is ground)
        lower_bound_max = [-e_s, -e_s, 0.0]    # X, Y, Z max
        upper_bound_min = [e_s, e_s, e_s]    # X, Y, Z min
        upper_bound_max = [e_s, e_s, e_s]      # X, Y, Z max

    class env_config:
        include_asset_type = {
            "bottom_wall": True,
        }

        asset_type_to_dict_map = {
            "bottom_wall": bottom_wall,
        }
