"""
Simple Hover SNN Task module for Aerial Gym Simulator.

A hover task where:
 - The quadrotor has access to an onboard IMU.
 - The environment consists of a ground plane only, with no obstacles.
 - The quadrotor must maintain a stable hover position using attitude commands.
 - No position information is provided in observations.

This task is identical to simple_hover - the SNN is only a different
neural network architecture used in training, not a different task.
"""

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import ssa # For angle wrapping
from aerial_gym.utils.logging import CustomLogger

import torch
import numpy as np
from gym.spaces import Dict, Box
from typing import Tuple, Dict as DictType
logger = CustomLogger("simple_hover_snn_task")


class HoverSNNTask(BaseTask):
    """
    Simple hover SNN task for quadrotor.
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
        Initialize the Simple Hover SNN Task.

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

        logger.info("Building environment for Simple Hover SNN Task...")
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

        # Target position bounds for randomization
        self.target_position_min = torch.tensor(
            self.task_config.target_position_min,
            device=self.device
        )
        self.target_position_max = torch.tensor(
            self.task_config.target_position_max,
            device=self.device
        )

        # Initialize random target positions for all envs
        self.target_position = torch.zeros(
            (self.num_envs, 3),
            device=self.device
        )
        self._randomize_targets(torch.arange(self.num_envs, device=self.device))

        logger.info(
            f"Simple Hover SNN Task initialized with {self.num_envs} environments."
            f" Observation space dim: {self.task_config.observation_space_dim},"
            f" Action space dim: {self.task_config.action_space_dim}"
        )

    def _randomize_targets(self, env_ids):
        """
        Randomize target positions for specified environments.
        Targets are uniformly sampled within [target_position_min, target_position_max].
        """
        num_resets = len(env_ids)
        random_values = torch.rand((num_resets, 3), device=self.device)
        self.target_position[env_ids] = (
            self.target_position_min +
            random_values * (self.target_position_max - self.target_position_min)
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
        # Randomize targets for all environments on full reset
        self._randomize_targets(torch.arange(self.num_envs, device=self.device))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        """
        Reset specific environments by their IDs.
        In this task - reset robot state and randomize target.
        """
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        # Randomize targets for reset environments
        self._randomize_targets(env_ids)
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

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

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
        Build 9D observation vector

        [0:3]   Position error to target (normalized by 5m)
        [3:6]   Body Linear Velocity (vx, vy, vz)
        [6:9]   Euler angles: roll, pitch, yaw (normalized)
        """

        # Position error (current position - target)
        position = self.obs_dict["robot_position"]
        position_error = position - self.target_position
        self.task_obs["observations"][:, 0:3] = position_error / 5.0  # Normalize by 5m

        # Velocity (body frame)
        self.task_obs["observations"][:, 3:6] = (
            self.obs_dict["robot_body_linvel"] / 5.0   # Normalize by max 5 m/s
        )

        # Euler angles (roll, pitch, yaw)
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        self.task_obs["observations"][:, 6:9] = euler_angles / torch.pi  # Normalize by pi

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        """
        Compute rewards and check for crashes.

        Rewards:
        - Exponential position reward: exp(-decay * ||pos_error||)
        - Hover bonus when ||pos_error|| < threshold
        - Small penalties for angular velocity and action jitter
        - Large penalty for collision

        Crashes:
        - Based on obs_dict["crashes"]

        Args:
            obs_dict: Dictionary of observations from the environment

        Returns:
            rewards: Tensor of shape (num_envs,) with computed rewards
            crashes: Tensor of shape (num_envs,) with crash flags (1 if crashed, 0 otherwise)
        """

        position_error = obs_dict["robot_position"] - self.target_position
        angular_velocity = obs_dict["robot_body_angvel"]
        crashes = obs_dict["crashes"]

        return compute_reward(
            position_error,
            angular_velocity,
            self.actions,
            self.prev_actions,
            crashes,
            self.task_config.reward_parameters
        )


@torch.jit.script
def compute_reward(
    position_error: torch.Tensor,
    angular_velocity: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    crashes: torch.Tensor,
    reward_params: DictType[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rewards for the hover task.

    Rewards:
    - Exponential position reward: exp(-decay * ||pos_error||)
    - Hover bonus when ||pos_error|| < threshold
    - Small penalties for angular velocity and action jitter
    - Large penalty for collision

    Args:
        position_error: Tensor of shape (num_envs, 3) with position error to target
        angular_velocity: Tensor of shape (num_envs, 3) with body angular velocities
        actions: Tensor of shape (num_envs, action_dim) with current actions
        prev_actions: Tensor of shape (num_envs, action_dim) with previous actions
        crashes: Tensor of shape (num_envs,) with crash flags (1 if crashed, 0 otherwise)
        reward_params: Dictionary with reward parameters

    Returns:
        rewards: Tensor of shape (num_envs,) with computed rewards
        crashes: Tensor of shape (num_envs,) with crash flags (1 if crashed, 0 otherwise)
    """

    pos_error_magnitude = torch.norm(position_error, dim=1)

    # Exponential position reward (1.0 at target, decays with distance)
    pos_reward = reward_params["pos_reward_scale"] * torch.exp(
        -reward_params["pos_reward_decay"] * pos_error_magnitude
    )

    # Hover bonus when close to target
    hover_bonus = torch.where(
        pos_error_magnitude < reward_params["hover_threshold"],
        reward_params["hover_bonus"],
        torch.zeros_like(pos_error_magnitude)
    )

    # Small penalties
    ang_vel_magnitude = torch.norm(angular_velocity, dim=1)
    angular_velocity_penalty = reward_params["angular_velocity_penalty"] * ang_vel_magnitude

    action_diff = torch.norm(actions - prev_actions, dim=1)
    jitter_penalty = reward_params["jitter_penalty"] * action_diff

    # Total reward
    rewards = pos_reward + hover_bonus - angular_velocity_penalty - jitter_penalty

    # Collision penalty
    rewards = torch.where(
        condition=crashes > 0,
        input=reward_params["collision_penalty"] * torch.ones_like(rewards),
        other=rewards
    )

    return rewards, crashes
