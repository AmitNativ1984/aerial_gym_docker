"""
Simple Hover Task module for Aerial Gym Simulator.

A hover task where:
 - The quadrotor has access to an onboard IMU.
 - The environment consists of a ground plane only, with no obstacles.
 - The quadrotor must maintain a stable hover position using attitude commands.
 - No position information is provided in observations.
"""

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import ssa # For angle wrapping
from aerial_gym.utils.logging import CustomLogger

import torch
import numpy as np
import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("simple_hover_task")


class HoverTask(BaseTask):
    """
    Simple hover task for quadrotor
    - Observation space: IMU readings (linear acceleration, angular velocity)
    - Action space: Attitude commands (roll, pitch, yaw_rate, thrust)
    
    Succeeds if the quadrotor maintains a stable hover.
    """

    def __init__(self, 
                 task_config,
                 seed=None,
                 num_envs=None,
                 headless=None,
                 device=None,
                 use_warp=None
    ):
        """
        Initialize the Simple Hover Task.

        Args:
            task_config: Task configuration class
            seed: Random seed for reproducibility (overrides task_config.seed if provided)
            num_envs: Number of parallel environments (overrides task_config.num_envs if provided)
            headless: Runs without rendering if True (overrides task_config.headless if provided)
            device: Device to run the simulation on (overrides task_config.device if provided)
            use_warp: Whether to use Warp for rendering (overrides task_config.use_warp if provided)
        """        
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        
        super().__init__(task_config)
        self.device = task_config.device

        # Convert reward parameters to tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key],
                device=self.device
            )

        logger.info("Building environment for Simple Hover Task...")
        logger.info(
            "\nSim Name: {}\nEnv Name: {}\nRobot Name: {}\nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name
            )
        )

        logger.info(
            "\nNum Envs: {}\nUse Warp: {}\nHeadless: {}\nDevice: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
                self.task_config.device
            )
        )

        # Build the simulation with SimBuilder
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.task_config.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless
        )

        # Initialize action tensors
        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False
        )
        
        self.prev_actions = torch.zeros_like(self.actions, device=self.device)

        # Get observations dictionary reference from environment
        # This dict is updated in-place by the sim step
        self.obs_dict = self.sim_env.get_obs()

        # Initialize tensors (rewards, terminations, truncations)
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        # Define observations Metadata for RL libraries
        # (Doesn't hold actual data - data is in obs_dict)
        self.observation_space = Dict(
            {"observations": Box(
                low=-1.0, 
                high=1.0, 
                shape=(self.task_config.observation_space_dim,),
                dtype=np.float32
                )
            }
        )
        
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32
        )

        self.action_transformation_function = self.task_config.action_transformation_function

        # Initialize task_obs dict
        self.task_obs = {
            "observations": torch.zeros(
                (self.task_config.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False
            ),
            "collisions": torch.zeros(
                (self.task_config.num_envs, 1),
                device=self.device,
                requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.task_config.num_envs, 1),
                device=self.device,
                requires_grad=False
            )
        }

        self.num_envs = self.sim_env.num_envs
        self.counter = 0
        # Info dictionary for additional info logging
        self.infos = {}

        logger.info(
            f"Simple Hover Task initialized with {self.num_envs} environments."
            f" Observation space dim: {self.task_config.observation_space_dim},"
            f" Action space dim: {self.task_config.action_space_dim}"
        )

    def close(self):
        """
        Clean up the environment and free resources.
        """
        self.sim_env.delete_env()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def reset(self):
        """
        Reset the task and environment.

        Returns:
            observations: Initial observations after reset
        """
        self.infos = {}
        self.sim_env.reset()
        return self.get_return_tuple()
    
    def reset_idx(self, env_ids):
        """
        Reset specific environments by their IDs.
        In this task - simply reset robot state.
        """
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        return

    def render(self):
        """
        Render the current state of the environment.
        """
        return None
    
    def step(self, actions):
        """
        Take a simulation step with the given actions.
        Does the following:
            - Save current actions to prev_actions before updating
            - Transform actions
            - Step the simulation
            - Compute rewards
            - Check for terminations/truncations
            - Handle resets if needed
        
        Returns:
            Tuple of (observations, rewards, terminations, truncations, info)
        
        
        """

        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        # Run the actions in the simulation environment
        transformed_actions = self.action_transformation_function(self.actions)
        self.sim_env.step(transformed_actions)

        # Calculate the rewards and check for terminations/truncations
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )

        self.sim_env.post_reward_calculation_step()

        self.infos = {}  # self.obs_dict["info"]

        if not self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        return return_tuple
    
    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos
        )
    
    def process_obs_for_task(self):
        """
        Build 12D observation vector

        [0:3]   Body Linear Velocity (vx, vy, vz)
        [3:6]   Euler angles: roll, pitch, yaw (normalized)
        [6:9]   Linear Acceleration from IMU (ax, ay, az)
        [9:12]  Angular Velocity from IMU (wx, wy, wz)
        """

        # Velocity (body frame)
        self.task_obs["observations"][:, 0:3] = (
            self.obs_dict["robot_body_linvel"] / 5.0   # Normalize by max 5 m/s
        )

        # Euler angles (roll, pitch, yaw)
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        self.task_obs["observations"][:, 3:6] = euler_angles / torch.pi  # Normalize by pi

        # Linear Acceleration from IMU
        self.task_obs["observations"][:, 6:9] = (
            self.obs_dict["imu_measurement"][:, 0:3] / 20.0  # Normalize by max 20 m/s^2 (gravity ~10 + motion ~10)
        )

        # Angular Velocity from IMU (normalize by ~10 rad/s)
        self.task_obs["observations"][:, 9:12] = (
            self.obs_dict["imu_measurement"][:, 3:6] / 10.0  # Normalize by max 10 rad/s
        )

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        """
        Compute rewards and check for crashes.

        Rewards:
        - Penalty for linear velocity
        - Penalty for angular velocity
        - Penalty for action jitter
        - Reward for maintaining hover
        - Penalty for collision

        Crashes:
        - Based on obs_dict["crashes"]

        Args:
            obs_dict: Dictionary of observations from the environment

        Returns:
            rewards: Tensor of shape (num_envs,) with computed rewards
            crashes: Tensor of shape (num_envs,) with crash flags (1 if crashed, 0 otherwise)
        """
        
        linear_velocity = obs_dict["robot_body_linvel"]
        angular_velocity = obs_dict["robot_body_angvel"]
        crashes = obs_dict["crashes"]

        return compute_reward(
            linear_velocity,
            angular_velocity,
            self.actions,
            self.prev_actions,
            crashes,
            self.task_config.reward_parameters
        )
    

@torch.jit.script
def compute_reward(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    crashes: torch.Tensor,
    reward_params: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rewards for the hover task.

    Args:
        linear_velocity: Tensor of shape (num_envs, 3) with body linear velocities
        angular_velocity: Tensor of shape (num_envs, 3) with body angular velocities
        actions: Tensor of shape (num_envs, action_dim) with current actions
        prev_actions: Tensor of shape (num_envs, action_dim) with previous actions
        crashes: Tensor of shape (num_envs,) with crash flags (1 if crashed, 0 otherwise)
        reward_params: Dictionary with reward parameters

    Returns:
        rewards: Tensor of shape (num_envs,) with computed rewards
        crashes: Tensor of shape (num_envs,) with crash flags (1 if crashed, 0 otherwise)
    """

    # Velocity penalties (penalize magnitude)
    lin_vel_magnitude = torch.norm(linear_velocity, dim=1)
    ang_vel_magnitude = torch.norm(angular_velocity, dim=1)

    velocity_penalty = reward_params["velocity_penalty"] * lin_vel_magnitude
    angular_velocity_penalty = reward_params["angular_velocity_penalty"] * ang_vel_magnitude

    # Jitter penalty (penalize large changes in actions)
    action_diff = torch.norm(actions - prev_actions, dim=1)
    jitter_penalty = reward_params["jitter_penalty"] * action_diff

    # Collision penalty
    collision_penalty = reward_params["collision_penalty"] * crashes

    # Total reward (all penalties)
    rewards = -velocity_penalty - angular_velocity_penalty - jitter_penalty 

    # Collision penalty
    rewards = torch.where(
        condition=crashes > 0,
        input=reward_params["collision_penalty"] * torch.ones_like(rewards),
        other=rewards
    )

    return rewards, crashes