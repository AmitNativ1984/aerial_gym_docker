"""
Visualize the Simple Obstacle Avoidance Environment.

Run with GUI to see the environment, drone, and obstacles.
Requires X11 forwarding or running in GUI mode.

Usage:
    # First enable X11 access (on host)
    xhost +local:docker

    # Then run this script
    python -m simple_obstacle_avoidance.tests.view_env
"""
# CRITICAL: isaacgym must be imported before torch
import isaacgym

import sys
sys.path.insert(0, "/workspaces/aerial_gym_docker")

import torch
import time

# Register our custom environment and task
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.task_registry import task_registry

from simple_obstacle_avoidance.task.simple_obstacle_avoidance_task import (
    SimpleObstacleAvoidanceTask,
)
from simple_obstacle_avoidance.config.task_config import task_config
from simple_obstacle_avoidance.config.env_config import SimpleObstacleEnvCfg

# Register environment and task
env_config_registry.register("simple_obstacle_env", SimpleObstacleEnvCfg)
task_registry.register_task(
    "simple_obstacle_avoidance_task",
    SimpleObstacleAvoidanceTask,
    task_config,
)


def main():
    print("=" * 50)
    print("Simple Obstacle Avoidance - Environment Viewer")
    print("=" * 50)

    # Create task with GUI enabled (headless=False)
    print("\nCreating environment with visualization...")
    task = task_registry.make_task(
        "simple_obstacle_avoidance_task",
        num_envs=4,        # Few envs for visualization
        headless=False,    # Enable GUI
        use_warp=True,
    )

    print(f"\nEnvironment created:")
    print(f"  - Num envs: {task.num_envs}")
    print(f"  - Observation dim: {task.task_config.observation_space_dim}")
    print(f"  - Action dim: {task.task_config.action_space_dim}")
    print(f"  - Obstacles: {task.task_config.curriculum.min_level}")
    print(f"  - Episode length: {task.task_config.episode_len_steps} steps")

    # Reset environment
    obs = task.reset()
    print("\nEnvironment reset. Running simulation loop...")
    print("Press Ctrl+C to exit\n")

    step = 0
    try:
        while True:
            # Random actions for visualization
            # Action space: [roll, pitch, yaw_rate, thrust] in [-1, 1]
            actions = torch.zeros((task.num_envs, 4), device=task.device)

            # Hover with small random perturbations
            actions[:, 0] = torch.randn(task.num_envs, device=task.device) * 0.1  # roll
            actions[:, 1] = torch.randn(task.num_envs, device=task.device) * 0.1  # pitch
            actions[:, 2] = torch.randn(task.num_envs, device=task.device) * 0.1  # yaw
            actions[:, 3] = 0.0  # Neutral thrust (transforms to hover)

            # Step environment
            obs, rewards, terminations, truncations, infos = task.step(actions)

            # Render
            task.render()

            step += 1
            if step % 100 == 0:
                crashes = infos.get("crashes", torch.zeros(1))
                print(
                    f"Step {step:5d} | "
                    f"Reward: {rewards.mean().item():7.3f} | "
                    f"Crashes: {crashes.sum().item():.0f}"
                )

            # Small delay for visualization
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        task.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
