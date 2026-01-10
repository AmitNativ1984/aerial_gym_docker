"""
Task configuration for simple hover task with onboard IMU.
Defines:
    - Observation space dimensions
    - Action space dimensions
    - Reward parameters
"""

import torch

class task_config:

    seed = 42  # Fixed random seed for reproducibility
    sim_name = "base_sim"
    env_name = "simple_hover_env"  # Our custom simple hover environment
    
    robot_name = "custom_quad_with_imu"  # Quadrotor with IMU
    controller_name = "lee_attitude_control"  # Attitude control
    args = {}

    # Environment settings
    num_envs = 512
    use_warp = False
    headless = False
    device = "cuda:0"

    privileged_observation_space_dim = 0
    
    # Observation space dim:
    # Velocity (3): [vx, vy, vz]
    # Orientation (3): [roll, pitch, yaw]
    # Linear Acceleration (3): [ax, ay, az]
    # Angular Velocity (3): [wx, wy, wz]
    observation_space_dim = 12

    # Action space dim: [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd]
    action_space_dim = 4

    # Episode length
    episode_len_steps = 1000 # real physics time for simulation is this value multiplied by sim.dt
    return_state_before_reset = False

    # Reward parameters
    reward_parameters = {
        "velocity_penalty": 0.1,            # Penalty for linear velocity
        "angular_velocity_penalty": 0.1,    # Penalty for angular velocity
        "jitter_penalty": 0.05,             # Penalty on  action jitter
        "collision_penalty": -100.0,          # Penalty for collision
    }

    @staticmethod
    def action_transformation_function(action):
        """
        Transform network output [-1,1] to attitude commands.

        Input: action tensor of shape (num_envs, 4), in range [-1, 1]
        Output: transformed action tensor for attitude controller
                - [0]: roll command [-pi/6, pi/6] rad
                - [1]: pitch command [-pi/6, pi/6] rad
                - [2]: yaw rate command [-pi/3, pi/3] rad/s
                - [3]: thrust command [0, 15] m/s^2
        """

        max_roll = torch.pi / 6  # Max roll/pitch in radians
        max_pitch = torch.pi / 6  # Max pitch in radians
        max_yaw_rate = torch.pi / 3     # Max yaw rate in radians/sec
        max_thrust = 15.0               # Max thrust in m/s^2
        
        clamped_action = torch.clamp(action, -1.0, 1.0) # Ensure action is in [-1, 1]

        roll_cmd = clamped_action[:, 0] * max_roll
        pitch_cmd = clamped_action[:, 1] * max_pitch
        yaw_rate_cmd = clamped_action[:, 2] * max_yaw_rate
        # Scale thrust from [-1, 1] to [0, max_thrust]
        thrust_cmd = (clamped_action[:, 3] + 1.0) / 2.0 * max_thrust  # Scale [0, max_thrust]

        transformed_action = torch.stack(
            [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd], dim=1
        )

        return transformed_action