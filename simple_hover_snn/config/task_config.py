"""
Task configuration for simple hover SNN task with onboard IMU.
Defines:
    - Observation space dimensions
    - Action space dimensions
    - Reward parameters

This is identical to simple_hover task config - the SNN is only
a different neural network architecture, not a different task.
"""

import torch

class task_config:

    seed = 42  # Fixed random seed for reproducibility
    sim_name = "base_sim"
    env_name = "simple_hover_snn_env"  # SNN variant environment

    robot_name = "custom_quad_with_imu_snn"  # Quadrotor with IMU (SNN variant)
    controller_name = "lee_attitude_control"  # Attitude control
    args = {}

    # Environment settings
    num_envs = 512
    use_warp = False
    headless = False
    device = "cuda:0"

    privileged_observation_space_dim = 0

    # Observation space dim:
    # Position error to target (3): [px - tx, py - ty, pz - tz]
    # Body Linear Velocity (3): [vx, vy, vz]
    # Euler angles (3): [roll, pitch, yaw]
    observation_space_dim = 9

    # Target hover position bounds (middle of environment ±20%)
    # Environment is ±5m XY, 0-5m Z. Middle ±20% of 5m = ±1m
    target_position_min = [-1.0, -1.0, 1.5]  # Min bounds for random target
    target_position_max = [1.0, 1.0, 3.5]    # Max bounds for random target

    # Action space dim (network output): [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd]
    # Note: action_transformation_function reorders to [thrust, roll, pitch, yaw_rate] for controller
    action_space_dim = 4

    # Episode length
    episode_len_steps = 1000 # real physics time for simulation is this value multiplied by sim.dt
    return_state_before_reset = False

    # Reward parameters
    reward_parameters = {
        "pos_reward_scale": 1.0,           # Scale for exponential position reward
        "pos_reward_decay": 2.0,           # Decay rate for exp(-decay * distance)
        "hover_bonus": 0.5,                # Bonus when within hover_threshold
        "hover_threshold": 0.1,            # Distance threshold for hover bonus (meters)
        "angular_velocity_penalty": 0.05,  # Small penalty for angular velocity
        "jitter_penalty": 0.01,            # Small penalty for action jitter
        "collision_penalty": -100.0,       # Large penalty for collision
    }

    @staticmethod
    def action_transformation_function(action):
        """
        Transform network output [-1,1] to attitude commands.

        Input: action tensor of shape (num_envs, 4), in range [-1, 1]
            - [0]: roll command
            - [1]: pitch command
            - [2]: yaw rate command
            - [3]: thrust command

        Output: transformed action tensor for LeeAttitudeController
            Controller expects: [thrust, roll, pitch, yaw_rate]
                - [0]: thrust command [-1, 1] (controller scales by mass*g)
                - [1]: roll command [-pi/6, pi/6] rad
                - [2]: pitch command [-pi/6, pi/6] rad
                - [3]: yaw rate command [-pi/3, pi/3] rad/s
        """

        max_roll = torch.pi / 6         # Max roll in radians (30 deg)
        max_pitch = torch.pi / 6        # Max pitch in radians (30 deg)
        max_yaw_rate = torch.pi / 3     # Max yaw rate in radians/sec (60 deg/s)

        clamped_action = torch.clamp(action, -1.0, 1.0)

        roll_cmd = clamped_action[:, 0] * max_roll
        pitch_cmd = clamped_action[:, 1] * max_pitch
        yaw_rate_cmd = clamped_action[:, 2] * max_yaw_rate
        # Thrust stays in [-1, 1] - controller applies: (thrust + 1) * mass * g
        # So -1 = 0 thrust, 0 = hover (1g), +1 = 2g
        thrust_cmd = clamped_action[:, 3]

        # Controller expects order: [thrust, roll, pitch, yaw_rate]
        transformed_action = torch.stack(
            [thrust_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd], dim=1
        )

        return transformed_action
