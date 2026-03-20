"""
Task configuration for navigation with obstacles.
Defines observation/action spaces, reward parameters, curriculum, and VAE settings.
"""
import torch


class task_config:
    """
    Configuration for NavigationWithObstaclesTask.

    Key features:
    - Acceleration control (accel_x, accel_y, accel_z, yaw_rate)
    - Custom 32D DepthVAE encoding
    - 30-level curriculum (panels then cumulative panels + objects)
    - Randomized environment bounds
    """

    seed = -1

    # Simulation components
    sim_name = "base_sim"
    env_name = "navigation_obstacle_env"
    robot_name = "nav_quadrotor_with_camera"
    controller_name = "lee_acceleration_control"
    args = {}

    # Environment settings
    num_envs = 1024
    use_warp = True
    headless = True
    device = "cuda:0"

    # Observation space: 14 (state) + 32 (VAE latents) = 46
    # State: [log_d_hor(1), d_z(1), d_norm(3), vel_w(3), angular_vel_b(3), angular_acc_b(3)]
    observation_space_dim = 14 + 32
    privileged_observation_space_dim = 0

    # Action space: [accel_x, accel_y, accel_z, yaw_rate]
    action_space_dim = 4

    # Action scaling: network outputs [-1, 1], scaled to physical units
    max_accel = 2.0              # m/s² per axis (symmetric: [-max, +max])
    max_yaw_rate = torch.pi / 3  # rad/s (~60 deg/s, symmetric: [-max, +max])

    # Episode length
    episode_len_steps = 200

    return_state_before_reset = False

    # Target waypoint placement (as ratio of environment bounds)
    # Target is placed in the far end of the environment
    target_min_ratio = [0.95, 0.10, 0.10]
    target_max_ratio = [1.00, 0.90, 0.90]

    # Reward parameters
    reward_parameters = {
        # Terminal rewards
        "exceed_penalty": -100.0,     # out-of-bounds termination
        "arrive_bonus": 100.0,        # reached target (success)
        "collision_penalty": -100.0,  # obstacle collision termination
        "d_min": 1.0,                 # arrival distance threshold (meters)
        # Progress reward (dense shaping, all lambda < 0)
        "lambda_d": -0.005,           # distance to target (horizontal + vertical)
        "lambda_v": -0.005,           # excessive horizontal speed penalty
        "lambda_dir": -0.003,         # velocity-goal direction misalignment
        "lambda_input": -0.0025,      # angular velocity penalty (attitude stability)
        "lambda_perc": -0.005,        # perception: discourage lateral/backward flight
    }

    # Speed threshold for excess speed penalty (m/s)
    v_max = 3.0

    class vae_config:
        """Custom 32D DepthVAE configuration."""
        use_vae = True
        latent_dims = 32

        # Path to trained DepthVAE checkpoint
        model_file = "/workspaces/aerial_gym_docker/vae_depth/runs/20260218_204641/checkpoints/epoch_150.pth"

        # DepthVAE input resolution
        target_height = 180
        target_width = 320

        # Depth range parameters
        max_depth_m = 7.0
        min_depth_m = 0.1
        sensor_max_range = 10.0

    class curriculum:
        """
        Curriculum configuration — same thresholds as original NavigationTask.
        Levels 0-5: large panels
        Levels 6-30: cumulative panels + small objects
        """
        min_level = 0
        max_level = 25
        check_after_log_instances = 2048
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6

    @staticmethod
    def action_transformation_function(action):
        """
        Transform network output [-1, 1] to acceleration commands.

        Scaling is driven by task_config.max_accel and task_config.max_yaw_rate.
        Input: action tensor (num_envs, 4) in range [-1, 1]
        Output: [accel_x, accel_y, accel_z, yaw_rate] for lee_acceleration_control
        """
        clamped_action = torch.clamp(action, -1.0, 1.0)

        processed = torch.zeros_like(clamped_action)
        processed[:, 0:3] = clamped_action[:, 0:3] * task_config.max_accel
        processed[:, 3] = clamped_action[:, 3] * task_config.max_yaw_rate

        return processed
