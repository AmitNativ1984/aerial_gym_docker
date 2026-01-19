"""
Custom runner for Hover task using RL Games framework.

The script registers the hover task and environment with aerial_gym and rl_games,
then runs PPO training.
"""

import isaacgym
import argparse
import os
import sys
import yaml
import numpy as np
sys.path.insert(0, "/workspaces/aerial_gym_docker")
import wandb
import torch
import gym
from gym import spaces
from loguru import logger
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots import BaseMultirotor
from aerial_gym.utils.helpers import parse_arguments
from distutils.util import strtobool

from simple_hover.task.simple_hover_task import HoverTask
from simple_hover.config.task_config import task_config
from simple_hover.config.env_config import SimpleHoverEnvCfg
from simple_hover.config.robot_config import CustomQuadWithImuCfg

# =============================================================================
# Register Custom Environment and Task
# =============================================================================


# Registe task with rl_games
env_config_registry.register("simple_hover_env", 
                             SimpleHoverEnvCfg)

# Register robot config
robot_registry.register("custom_quad_with_imu", 
                        BaseMultirotor,
                        CustomQuadWithImuCfg)
# Register task
task_registry.register_task("simple_hover_task", 
                       HoverTask,
                       task_config)

# =============================================================================
# RL Games Integration
# =============================================================================

class ExtractObsWrapper(gym.Wrapper):
    """
    Wrapper that extracts the 'observations' tensor from the observation dict.
    rl_games expects flat observation tensors, not dicts.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations["observations"]
    
    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        
        dones = torch.where(
            terminated | truncated, 
            torch.ones_like(terminated),
            torch.zeros_like(terminated)
        )

        return observations["observations"], rewards, dones, infos
    
class AERIALRLGPUEnv(vecenv.IVecEnv):
    """Vectorized environment wrapper for rl_games."""

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](
            **kwargs
        )
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)
    
    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()
    
    def get_number_of_agents(self):
        return self.env.get_number_of_agents()
    
    def get_env_info(self):
        """Return observation and action space info for rl_games."""

        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
             np.ones(self.env.task_config.action_space_dim)
        )
        info["observation_space"] = spaces.Box(
            -np.ones(self.env.task_config.observation_space_dim) * np.inf,
             np.ones(self.env.task_config.observation_space_dim) * np.inf
        )
        return info
    

# Register task with rl_games env_configurations
env_configurations.register(
    "simple_hover_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("simple_hover_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

# Register vectorized environment type
vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs)
)

# =============================================================================
# Argument Parsing
# =============================================================================


def get_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {"name": "--train", "action": "store_true", "help": "Train network"},
        {"name": "--play", "action": "store_true", "help": "Play/test network"},
        {"name": "--checkpoint", "type": str, "help": "Path to checkpoint"},
        {"name": "--file", "type": str, "default": "...", "help": "Path to config"},
        {"name": "--num_envs", "type": int, "default": 1024, "help": "Num envs"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": "False", "help": "Headless mode"},
        {"name": "--use_warp", "type": lambda x: bool(strtobool(x)), "default": "False", "help": "Use warp"},
        {"name": "--experiment_name", "type": str, "default": "simple_hover", "help": "Experiment name"},
        {"name": "--task", "type": str, "default": "simple_hover_task", "help": "Task name"},
        {"name": "--track", "action": "store_true", "help": "Track with Weights and Biases"},
        {"name": "--wandb-project-name", "type": str, "default": "aerial_gym", "help": "Wandb project name"},
        {"name": "--wandb-entity", "type": str, "default": None, "help": "Wandb entity (team)"},
    ]
    args = parse_arguments(description="Simple Hover Task", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args

def update_config(config, args):
    """Update training config with CLI arguments."""
    config["params"]["config"]["env_name"] = args["task"]
    config["params"]["config"]["name"] = args["experiment_name"]
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]

    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]

    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Create output directories
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Parse arguments
    args = vars(get_args())
    config_name = args["file"]

    logger.info(f"Loading config: {config_name}")
    logger.info(f"Number of environments: {args['num_envs']}")
    logger.info(f"Headless: {args['headless']}")
    logger.info(f"Use warp: {args['use_warp']}")

    # Load and update config
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)
        config = update_config(config, args)

        # Initialize rl_games runner
        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading config: {exc}")
            sys.exit(1)

    # Initialize wandb tracking if enabled
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args["track"] and rank == 0:
        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )

    # Run training or playing
    logger.info("Starting training..." if args.get("train") else "Starting playback...")
    runner.run(args)

    # Finish wandb tracking
    if args["track"] and rank == 0:
        wandb.finish()

    logger.info("Done!")
