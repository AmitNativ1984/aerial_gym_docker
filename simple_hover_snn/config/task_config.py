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

    robot_name = "base_quadrotor"  # Base quadrotor (matching position_setpoint_task)
    controller_name = "lee_attitude_control"  # Attitude control
    args = {}

    # Environment settings
    num_envs = 4096  # Matching position_setpoint_task
    use_warp = False
    headless = False
    device = "cuda:0"

    privileged_observation_space_dim = 0

    # Observation space dim (matching position_setpoint_task):
    # Position error to target (3): [tx - px, ty - py, tz - pz]
    # Robot orientation (4): quaternion [qx, qy, qz, qw]
    # Body Linear Velocity (3): [vx, vy, vz]
    # Body Angular Velocity (3): [wx, wy, wz]
    observation_space_dim = 13

    # Action space dim (network output): [thrust_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd]
    # Matches LeeAttitudeController expected format directly
    action_space_dim = 4

    # Episode length
    episode_len_steps = 800  # 30 seconds per episode (600 * 0.05s)
    return_state_before_reset = False

    # Success condition: hover at target for 3 seconds
    success_threshold = 0.10      # Distance threshold (meters) - 10cm
    success_hold_duration = 3.0   # Time to hold position (seconds)
    success_hold_steps = 60       # = success_hold_duration / env_step_dt (0.05s)

    # Reward parameters - Potential-based reward
    # R_total = R_progress + R_velocity + R_jitter + R_success + R_crash
    #
    # Components:
    # 1. Progress: k_progress * (prev_dist - curr_dist) - rewards approaching goal
    # 2. Gated velocity: -k_vel * g(dist) * ||v|| - penalizes velocity near goal
    #    g(dist) = sigmoid((gate_center - dist) / gate_width) - smooth 0→1 ramp
    # 3. Jitter: -k_jitter * ||action_diff|| - penalizes rapid action changes
    # 4. Success: +k_success per step inside threshold - continuous positive reward
    # 5. Crash: -k_crash - large penalty for exceeding max_distance
    reward_parameters = {
        # Crash penalty
        "k_crash": [50.0],            # Crash penalty magnitude
        "max_distance": [6.0],        # Distance beyond which robot crashes

        # Distance penalty: -k_dist * curr_dist (encourages getting closer)
        "k_dist": [1.0],              # Distance penalty coefficient

        # Progress reward: k_progress * (prev_dist - curr_dist)
        "k_progress": [10.0],         # Reward for moving toward goal (increased for faster progress)

        # Gated velocity penalty: -k_vel * g(dist) * ||v||
        # g(dist) = sigmoid((gate_center - dist) / gate_width)
        "gate_center": [0.5],         # Distance at which gate is 0.5 (meters)
        "gate_width": [0.1],          # Controls steepness of sigmoid ramp

        # Action jitter penalty: -k_jitter * ||action_diff||
        "k_jitter": [0.5],            # Jitter penalty coefficient

        # Attitude penalty: -k_tilt * g(dist) * (pitch^2 + roll^2)
        # Penalizes tilt when near goal (gated by sigmoid)
        "k_tilt": [1.0],              # Tilt penalty coefficient

        # Angular rate penalty: -k_angvel * g(dist) * ||angvel||
        # Penalizes rotation when near goal (gated by sigmoid)
        "k_angvel": [0.5],            # Angular velocity penalty coefficient

        # Hover bonus: +k_hover * exp(-||vel|| / vel_scale) - reward for stable hovering
        # Active when dist < threshold_hover, bonus decays exponentially with velocity
        "k_hover": [5.0],             # Hover bonus coefficient (must dominate other rewards)
        "threshold_hover": [0.2],     # Distance threshold for hover (meters) - 20cm
        "vel_scale_hover": [0.1],     # Velocity decay scale (m/s) - smaller = stricter
    }
