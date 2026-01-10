#!/bin/bash
#
# Training script for Simple Obstacle Avoidance Task
#
# Usage:
#   ./train_obstacle_avoidance.sh [num_envs]
#
# Examples:
#   ./train_obstacle_avoidance.sh          # Default 512 envs
#   ./train_obstacle_avoidance.sh 1024     # 1024 envs
#   ./train_obstacle_avoidance.sh 256      # 256 envs (for low VRAM)
#

set -e

cd /workspaces/aerial_gym_docker

NUM_ENVS=${1:-128}

echo "=========================================="
echo "Simple Obstacle Avoidance Training"
echo "=========================================="
echo "Number of environments: $NUM_ENVS"
echo "Config: simple_obstacle_avoidance/training/ppo_simple_obstacle.yaml"
echo ""

python -m simple_obstacle_avoidance.training.runner \
    --file=simple_obstacle_avoidance/training/ppo_simple_obstacle.yaml \
    --num_envs=$NUM_ENVS \
    --headless=True \
    --train

echo ""
echo "Training complete!"
