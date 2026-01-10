"""
Task configuration for simple obstacle avoidance.
Defines observation/action spaces, reward parameters, and VAE settings.
"""
import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config:
    """
    Configuration for SimpleObstacleAvoidanceTask.

    Key features:
    - Attitude control (roll, pitch, yaw_rate, thrust)
    - Depth camera with VAE encoding
    - Simplified reward function
    - No curriculum (fixed 5 obstacles)
    """

    # Random seed (-1 for random)
    seed = -1

    # Simulation components
    sim_name = "base_sim"
    env_name = "simple_obstacle_env"  # Our custom sparse obstacle env
    robot_name = "custom_quadrotor_with_camera"  # Custom spawn position + depth camera
    controller_name = "lee_attitude_control"  # Attitude control
    args = {}

    # Environment settings
    num_envs = 1024
    use_warp = True
    headless = True
    device = "cuda:0"

    # Observation space: 13 (state) + 64 (VAE latents) = 77
    # State: [unit_vec_to_target(3), dist(1), roll(1), pitch(1), reserved(1),
    #         body_linvel(3), body_angvel(3)]
    observation_space_dim = 13 + 64
    privileged_observation_space_dim = 0

    # Action space: [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd]
    action_space_dim = 4

    # Episode length (longer than navigation task for easier learning)
    episode_len_steps = 300

    # Return observation before or after reset
    return_state_before_reset = False

    # Target waypoint placement (as ratio of environment bounds)
    # Target is placed in the far end of the environment
    target_min_ratio = [0.80, 0.20, 0.20]
    target_max_ratio = [0.95, 0.80, 0.80]

    # Reward parameters (simplified from navigation task)
    reward_parameters = {
        "pos_reward_magnitude": 5.0,     # Reward for being close to target
        "pos_reward_exponent": 0.5,      # Exponential decay rate
        "getting_closer_reward": 5.0,    # Reward for progress toward target
        "collision_penalty": -100.0,     # Penalty for crashing
        "action_penalty": 0.01,          # Small penalty for large actions
    }

    class vae_config:
        """VAE configuration for depth image encoding."""
        use_vae = True
        latent_dims = 64

        # Use pre-trained VAE weights from aerial_gym
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        model_folder = AERIAL_GYM_DIRECTORY

        # Image resolution for VAE input
        image_res = (270, 480)
        interpolation_mode = "nearest"
        return_sampled_latent = True

    class curriculum:
        """
        Curriculum configuration - DISABLED for simple task.
        Fixed at 10 obstacles, never changes.
        """
        min_level = 10
        max_level = 10  # Same as min - no curriculum
        check_after_log_instances = 9999999  # Never check
        increase_step = 0
        decrease_step = 0
        success_rate_for_increase = 1.0  # Impossible to reach
        success_rate_for_decrease = 0.0  # Impossible to reach

    @staticmethod
    def action_transformation_function(action):
        """
        Transform network output [-1, 1] to attitude commands.

        Input: action tensor of shape (num_envs, 4) in range [-1, 1]
        Output: transformed action tensor for attitude controller
            - [0]: roll command [-pi/6, pi/6] rad
            - [1]: pitch command [-pi/6, pi/6] rad
            - [2]: yaw rate command [-pi/3, pi/3] rad/s
            - [3]: thrust command [0, 15] m/s^2
        """
        clamped_action = torch.clamp(action, -1.0, 1.0)

        # Maximum values for each action dimension
        max_roll = torch.pi / 6      # 30 degrees
        max_pitch = torch.pi / 6     # 30 degrees
        max_yaw_rate = torch.pi / 3  # 60 deg/s
        max_thrust = 15.0            # m/s^2 (roughly 1.5g)

        # Transform actions
        processed = torch.zeros_like(clamped_action)
        processed[:, 0] = clamped_action[:, 0] * max_roll      # Roll: symmetric
        processed[:, 1] = clamped_action[:, 1] * max_pitch     # Pitch: symmetric
        processed[:, 2] = clamped_action[:, 2] * max_yaw_rate  # Yaw rate: symmetric
        # Thrust: shifted from [-1,1] to [0, max_thrust]
        processed[:, 3] = (clamped_action[:, 3] + 1.0) * max_thrust / 2.0

        return processed
