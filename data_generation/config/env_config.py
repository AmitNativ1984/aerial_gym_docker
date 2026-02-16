from aerial_gym.config.asset_config.env_object_config import (
    panel_asset_params,
    thin_asset_params,
    tree_asset_params,
    object_asset_params,
)

import numpy as np


# Panels with full rotation randomization (replaces fixed walls)
class dense_panel_params(panel_asset_params):
    num_assets = 15
    keep_in_env = False  # allow density randomization to cull some

    # Full position range across the environment
    min_state_ratio = [
        0.05, 0.05, 0.05,
        -np.pi, -np.pi / 2, -np.pi,
        1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]
    max_state_ratio = [
        0.95, 0.95, 0.95,
        np.pi, np.pi / 2, np.pi,
        1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]


class dense_thin_params(thin_asset_params):
    num_assets = 8


class dense_tree_params(tree_asset_params):
    num_assets = 4


class dense_object_params(object_asset_params):
    num_assets = 50


class DataGenEnvCfg:
    class env:
        num_envs = 16  # default, overridden by CLI
        num_env_actions = 4
        env_spacing = 5.0

        num_physics_steps_per_env_step_mean = 1  # minimal physics needed
        num_physics_steps_per_env_step_std = 0

        render_viewer_every_n_steps = 1000  # effectively never
        reset_on_collision = False  # don't reset when camera is inside obstacle
        collision_force_threshold = 1000.0
        create_ground_plane = False
        sample_timestep_for_latency = False
        perturb_observations = False
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False

        use_warp = True

        # Environment bounds (randomized per-env on each reset)
        # Resulting volume: ~18-25m (X) x 14-20m (Y) x 8-14m (Z)
        lower_bound_min = [-5.0, -10.0, -7.0]
        lower_bound_max = [-3.0, -7.0, -4.0]
        upper_bound_min = [15.0, 7.0, 4.0]
        upper_bound_max = [20.0, 10.0, 7.0]

    class env_config:
        include_asset_type = {
            "panels": True,
            "thin": True,
            "trees": True,
            "objects": True,
        }

        asset_type_to_dict_map = {
            "panels": dense_panel_params,
            "thin": dense_thin_params,
            "trees": dense_tree_params,
            "objects": dense_object_params,
        }
