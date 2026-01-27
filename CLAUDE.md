# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Docker-based environment for the Aerial Gym Simulator, built on NVIDIA Isaac Gym Preview 4. Used for GPU-accelerated physics simulation and reinforcement learning training for aerial robotics.

## References
When reasoning about the code, you MUST consult the following online documentation:

- Local Aerial Gym Simulator:
  /app/aerial_gym/aerial_gym_simulator/
  (You can search this always)

- Isaac Gym API:
  https://github.com/isaac-sim/IsaacGymEnvs 
  https://developer.nvidia.com/isaac-gym

- Aerial Gym Simulator: 
  https://ntnu-arl.github.io/aerial_gym_simulator/
  https://github.com/emNavi/AirGym

- SnnTorch:
  https://snntorch.readthedocs.io/en/latest/

- Norse
  https://norse.github.io/notebooks/intro_norse.html
      

Use these references to:
1. Validate architecture choices
2. Match expected tensor shapes
3. Avoid deprecated APIs

If something in the local code conflicts with the docs, point it out explicitly.

**Tech Stack:** Python 3.6-3.8, PyTorch, NVIDIA Isaac Gym, Docker with NVIDIA runtime

## Common Commands

### Setup & Build
```bash
# Run full setup (checks prerequisites, extracts Isaac Gym, optionally builds)
./setup.sh

# Build Docker image
docker compose build
# OR
docker build -t aerial-gym:latest .
```

### Running Containers
```bash
# Headless mode (training)
docker compose run aerial-gym

# GUI mode (visualization)
xhost +local:docker && docker compose run aerial-gym-gui
# OR use helper script
./run.sh
```

### Testing Inside Container
```bash
# Test Isaac Gym
cd /opt/isaacgym/python/examples && python joint_monkey.py --headless

# Test Aerial Gym
cd /workspace/aerial_gym_simulator
python examples/position_control_example.py --headless

# Run RL training
python aerial_gym/rl_training/rl_games/runner.py \
    --file=./ppo_aerial_quad_position.yaml \
    --num_envs=512 --headless=True

# Navigation with depth images
python aerial_gym/rl_training/rl_games/runner.py \
    --file=./ppo_aerial_quad_navigation.yaml \
    --num_envs=256 --headless=True
```

## Architecture

### Docker Build (Dockerfile.base)
Multi-stage build from `nvcr.io/nvidia/pytorch:22.12-py3`:
1. **base**: System dependencies, Mesa EGL removal (critical fix for rendering crashes)
2. **isaacgym**: Isaac Gym installation at `/opt/isaacgym`, NVIDIA vendor configs
3. **aerial_gym**: Clones and installs aerial_gym_simulator from ntnu-arl/aerial_gym_simulator

### Key Paths (Inside Container)
- `/opt/isaacgym/` - Isaac Gym installation
- `/workspace/aerial_gym_simulator/` - Aerial Gym Simulator
- `/opt/isaacgym/python/examples/` - Isaac Gym example scripts

### Volume Mounts
- `./outputs` → `/workspace/aerial_gym_simulator/outputs` (training outputs)
- `./logs` → `/workspace/aerial_gym_simulator/logs` (logs)

## Critical Configuration Notes

1. **Isaac Gym package required**: Must manually download `IsaacGym_Preview_4_Package.tar.gz` from NVIDIA (requires account)

2. **Mesa EGL workaround**: Dockerfile removes Mesa EGL libraries to prevent segfaults. Never reinstall `libEGL_mesa.so`

3. **OpenCV version**: Pinned to 4.5.5.64 to avoid `cv2.dnn.DictValue` errors

4. **VRAM limits**: For 8GB GPUs, use `--num_envs=256` or lower

5. **X11 for GUI**: Requires `xhost +local:docker` before running GUI mode

## Isaac Gym API (for reference)
```python
from isaacgym import gymapi, gymtorch, gymutil
# gymapi - Main simulation API
# gymtorch - PyTorch tensor integration
# gymutil - Argument parsing, helpers
```
