"""
Simple Obstacle Avoidance Task for Aerial Gym Simulator.

A simplified navigation task where a quadrotor must:
1. Navigate to a target waypoint
2. Avoid sparse obstacles using depth camera observations
3. Use attitude control (roll, pitch, yaw_rate, thrust)

Based on the NavigationTask but with:
- Fewer obstacles (5 fixed, no curriculum)
- Simpler reward function
- Attitude control instead of velocity control
"""
from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import (
    quat_rotate_inverse,
    torch_rand_float_tensor,
    torch_interpolate_ratio,
    ssa,
)
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

import torch
import numpy as np
import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("simple_obstacle_avoidance_task")


class SimpleObstacleAvoidanceTask(BaseTask):
    """
    Simple obstacle avoidance task with:
    - Sparse obstacles (5 fixed)
    - Depth camera + VAE encoding
    - Attitude control (roll, pitch, yaw_rate, thrust)
    - Navigate to waypoint
    """

    def __init__(
        self,
        task_config,
        seed=None,
        num_envs=None,
        headless=None,
        device=None,
        use_warp=None,
    ):
        """
        Initialize the simple obstacle avoidance task.

        Args:
            task_config: Task configuration class
            seed: Random seed (overrides config if provided)
            num_envs: Number of parallel environments (overrides config)
            headless: Run without visualization (overrides config)
            device: CUDA device (overrides config)
            use_warp: Use warp rendering (overrides config)
        """
        # Override config params if provided
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
        self.device = self.task_config.device

        # Convert reward params to tensors for efficient GPU computation
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        logger.info("Building Simple Obstacle Avoidance environment")
        logger.info(
            f"Sim: {task_config.sim_name}, Env: {task_config.env_name}, "
            f"Robot: {task_config.robot_name}, Controller: {task_config.controller_name}"
        )

        # Build simulation environment using SimBuilder
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        # Target position for each environment
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Target sampling ratios (as fraction of environment bounds)
        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device
        ).expand(self.sim_env.num_envs, -1)

        # Previous distance to target (for progress reward)
        self.prev_dist = torch.zeros(self.sim_env.num_envs, device=self.device)

        # VAE encoder for depth images
        if self.task_config.vae_config.use_vae:
            self.vae_model = VAEImageEncoder(
                config=self.task_config.vae_config, device=self.device
            )
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = None
            self.image_latents = None

        # Get observation dictionary reference from environment
        # This dict is updated in-place by the simulator
        self.obs_dict = self.sim_env.get_obs()

        # Set fixed number of obstacles (no curriculum)
        self.obs_dict["num_obstacles_in_env"] = self.task_config.curriculum.min_level

        # References to termination/truncation tensors
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.sim_env.num_envs, device=self.device)

        # Define observation and action spaces for rl_games
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Action transformation function
        self.action_transformation_function = (
            self.task_config.action_transformation_function
        )

        # Number of environments
        self.num_envs = self.sim_env.num_envs

        # Task observation tensor (filled each step)
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
        }

        # Info dictionary for logging
        self.infos = {}

        # Episode counter per environment for keep_same_env_for_num_episodes
        self.episode_counter = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        # Get the threshold from env config (default to 1 if not set)
        self.keep_same_env_episodes = getattr(
            self.sim_env.cfg.env, 'keep_same_env_for_num_episodes', 1
        )

        logger.info(
            f"Task initialized with {self.num_envs} environments, "
            f"obs_dim={self.task_config.observation_space_dim}, "
            f"action_dim={self.task_config.action_space_dim}"
        )

    def close(self):
        """Clean up simulation resources."""
        # Clean up CUDA memory
        del self.sim_env
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def reset(self):
        """
        Reset all environments.

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        # Force obstacle reset on full reset (e.g., at start of training)
        self.reset_idx(
            torch.arange(self.sim_env.num_envs, device=self.device),
            force_obstacle_reset=True
        )
        return self.get_return_tuple()

    def reset_idx(self, env_ids, force_obstacle_reset=False):
        """
        Reset specific environments.

        Args:
            env_ids: Tensor of environment indices to reset
            force_obstacle_reset: If True, reset obstacles regardless of episode counter
        """
        if len(env_ids) == 0:
            return

        # Increment episode counter for these environments
        self.episode_counter[env_ids] += 1

        # Determine which environments need full obstacle reset
        if force_obstacle_reset:
            envs_needing_obstacle_reset = env_ids
        else:
            # Only reset obstacles when episode counter reaches threshold
            needs_reset_mask = self.episode_counter[env_ids] >= self.keep_same_env_episodes
            envs_needing_obstacle_reset = env_ids[needs_reset_mask]

        # Reset episode counter for envs that will get new obstacles
        if len(envs_needing_obstacle_reset) > 0:
            self.episode_counter[envs_needing_obstacle_reset] = 0

        # Call sim_env reset only for environments that need obstacle re-randomization
        # For others, just reset robot position (handled by robot_manager internally)
        if len(envs_needing_obstacle_reset) > 0:
            # Full reset including obstacles
            self.sim_env.reset_idx(envs_needing_obstacle_reset)

        # Reset robot only for remaining environments (no obstacle change)
        envs_robot_only = env_ids[~torch.isin(env_ids, envs_needing_obstacle_reset)] if len(envs_needing_obstacle_reset) > 0 else env_ids
        if len(envs_robot_only) > 0:
            # Only reset robot, not obstacles
            self.sim_env.robot_manager.reset_idx(envs_robot_only)
            self.sim_env.IGE_env.write_to_sim()
            self.sim_env.sim_steps[envs_robot_only] = 0

        # Sample new target positions within bounds
        target_ratio = torch_rand_float_tensor(
            self.target_min_ratio, self.target_max_ratio
        )
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )

        # Reset previous distance for progress reward
        self.prev_dist[env_ids] = torch.norm(
            self.target_position[env_ids] - self.obs_dict["robot_position"][env_ids],
            dim=1,
        )

        # Clear info dict
        self.infos = {}

    def render(self):
        """Render the environment."""
        return self.sim_env.render()

    def step(self, actions):
        """
        Execute one step of the simulation.

        Args:
            actions: Tensor of actions (num_envs, 4) in range [-1, 1]

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        # Transform network outputs to controller commands
        transformed_action = self.action_transformation_function(actions)

        # Step the simulation
        self.sim_env.step(actions=transformed_action)

        # Compute rewards and detect crashes
        self.rewards[:], self.terminations[:] = self.compute_rewards(self.obs_dict)

        # Check for episode timeout (truncation)
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # Track successes, crashes, and timeouts for logging
        dist_to_target = torch.norm(
            self.target_position - self.obs_dict["robot_position"], dim=1
        )

        # Success: reached target without crashing
        successes = self.truncations * (dist_to_target < 1.0)
        successes = torch.where(
            self.terminations > 0, torch.zeros_like(successes), successes
        )

        # Timeout: truncated but didn't reach target
        timeouts = torch.where(
            self.truncations > 0,
            torch.logical_not(successes),
            torch.zeros_like(successes),
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        # Handle resets for terminated/truncated environments
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        # Process depth image through VAE encoder
        self.process_image_observation()

        return self.get_return_tuple()

    def process_image_observation(self):
        """Encode depth image through VAE to get latent representation."""
        if self.task_config.vae_config.use_vae and self.vae_model is not None:
            # Get depth image: (num_envs, 1, H, W) -> (num_envs, H, W)
            image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
            # Encode to latent space
            self.image_latents[:] = self.vae_model.encode(image_obs)

    def get_return_tuple(self):
        """
        Build and return the step/reset output tuple.

        Returns:
            Tuple of (task_obs, rewards, terminations, truncations, infos)
        """
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        """
        Build observation vector (77D total).

        Observation structure:
        - [0:3]   Unit vector to target (vehicle frame)
        - [3]     Distance to target (normalized)
        - [4]     Roll angle (normalized)
        - [5]     Pitch angle (normalized)
        - [6]     Reserved (0)
        - [7:10]  Body linear velocity (normalized)
        - [10:13] Body angular velocity (normalized)
        - [13:77] VAE latent encoding (64D depth features)
        """
        # Compute vector to target in vehicle frame
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)
        unit_vec_to_tgt = vec_to_tgt / (dist_to_tgt.unsqueeze(1) + 1e-6)

        # [0:3] Unit vector to target in vehicle frame
        self.task_obs["observations"][:, 0:3] = unit_vec_to_tgt

        # [3] Distance to target (normalized by max expected distance)
        self.task_obs["observations"][:, 3] = dist_to_tgt / 20.0

        # [4:6] Roll and pitch angles (normalized by pi)
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        self.task_obs["observations"][:, 4] = euler_angles[:, 0] / torch.pi  # Roll
        self.task_obs["observations"][:, 5] = euler_angles[:, 1] / torch.pi  # Pitch

        # [6] Reserved
        self.task_obs["observations"][:, 6] = 0.0

        # [7:10] Body linear velocity (normalized)
        self.task_obs["observations"][:, 7:10] = (
            self.obs_dict["robot_body_linvel"] / 5.0
        )

        # [10:13] Body angular velocity (normalized)
        self.task_obs["observations"][:, 10:13] = (
            self.obs_dict["robot_body_angvel"] / 5.0
        )

        # [13:77] VAE latent encoding (depth features)
        if self.task_config.vae_config.use_vae and self.image_latents is not None:
            self.task_obs["observations"][:, 13:] = self.image_latents

    def compute_rewards(self, obs_dict):
        """
        Compute rewards for all environments.

        Reward components:
        - Position reward: Exponential reward based on distance to target
        - Progress reward: Bonus for getting closer to target
        - Collision penalty: Large negative reward for crashing

        Args:
            obs_dict: Dictionary containing environment observations

        Returns:
            Tuple of (rewards, crashes) tensors
        """
        robot_pos = obs_dict["robot_position"]
        crashes = obs_dict["crashes"]

        # Compute distance to target
        dist = torch.norm(self.target_position - robot_pos, dim=1)

        # Position reward: exponential decay with distance
        # Higher reward when closer to target
        pos_reward = self.task_config.reward_parameters[
            "pos_reward_magnitude"
        ] * torch.exp(-dist * self.task_config.reward_parameters["pos_reward_exponent"])

        # Progress reward: reward for getting closer
        progress = self.prev_dist - dist
        getting_closer_reward = (
            self.task_config.reward_parameters["getting_closer_reward"] * progress
        )

        # Update previous distance for next step
        self.prev_dist[:] = dist

        # Total reward
        reward = pos_reward + getting_closer_reward

        # Apply collision penalty (overrides other rewards)
        reward = torch.where(
            crashes > 0,
            self.task_config.reward_parameters["collision_penalty"]
            * torch.ones_like(reward),
            reward,
        )

        return reward, crashes
