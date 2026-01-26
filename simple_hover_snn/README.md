# Simple Hover SNN Task

A quadrotor hover task using a Spiking Neural Network (SNN) for control policy.

## Task Overview

The drone must navigate from a random starting position to hover at a fixed target point `(0, 0, 2.5)` in the center of the environment.

---

## Environment

| Property | Value |
|----------|-------|
| Bounds | 10m × 10m × 5m (X: ±5m, Y: ±5m, Z: 0-5m) |
| Ground plane | Yes |
| Obstacles | None |
| Out-of-bounds | Episode terminates |
| Collision | Episode terminates |

---

## Drone & Sensors

### Quadrotor
- Standard quadrotor with attitude controller (Lee attitude control)
- 4 rotors, symmetric configuration

### Sensors
| Sensor | Measurements |
|--------|--------------|
| IMU (Bosch BMI088) | Linear acceleration (3D), Angular velocity (3D) |
| State estimator | Position (3D), Velocity (3D), Euler angles (3D) |

### Spawn Configuration
| Property | Range |
|----------|-------|
| Position X, Y | -2m to +2m (ratio 0.3-0.7 of ±5m) |
| Position Z | 1.5m to 3.5m (ratio 0.3-0.7 of 5m) |
| Roll, Pitch | ±30° (±π/6 rad) |
| Yaw | 0° to 360° |
| Linear velocity | ±1 m/s (X,Y), ±0.5 m/s (Z) |
| Angular velocity | ±0.5 rad/s |

---

## Observations (15D)

| Index | Observation | Normalization |
|-------|-------------|---------------|
| 0-2 | Position error to target (x, y, z) | ÷ 5.0 m |
| 3-5 | Body linear velocity (vx, vy, vz) | ÷ 5.0 m/s |
| 6-8 | Euler angles (roll, pitch, yaw) | ÷ π rad |
| 9-11 | IMU linear acceleration (ax, ay, az) | ÷ 20.0 m/s² |
| 12-14 | IMU angular velocity (ωx, ωy, ωz) | ÷ 10.0 rad/s |

---

## Actions (4D)

Network outputs in range `[-1, 1]`, transformed to attitude commands:

| Index | Action | Transform | Physical Range |
|-------|--------|-----------|----------------|
| 0 | Roll command | × π/6 | ±30° |
| 1 | Pitch command | × π/6 | ±30° |
| 2 | Yaw rate command | × π/3 | ±60°/s |
| 3 | Thrust command | [0, 15] m/s² | 0-15 m/s² |

---

## Reward Function

All rewards are penalties (negative):

| Component | Weight | Formula |
|-----------|--------|---------|
| Position error | 1.0 | `‖position - target‖` |
| Linear velocity | 0.1 | `‖velocity‖` |
| Angular velocity | 0.1 | `‖angular_velocity‖` |
| Action jitter | 0.05 | `‖action - prev_action‖` |
| Collision | -100.0 | Fixed penalty on crash |

**Total reward:** `-(pos_penalty + vel_penalty + ang_vel_penalty + jitter_penalty)`

---

## Neural Network (SNN)

### Architecture
Spiking Neural Network with Leaky Integrate-and-Fire (LIF) neurons.

```
                    ┌─────────────────────────────────────┐
                    │         Policy Network              │
                    │  Input(15) → LIF(64) → LIF(64) → LIF(64)
                    │                 ↓                   │
                    │         action_head(4) → μ          │
                    │         log_std(4) → σ              │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         Value Network               │
                    │  Input(15) → LIF(64) → LIF(64) → LIF(64)
                    │                 ↓                   │
                    │         value_head(1) → V           │
                    └─────────────────────────────────────┘
```

### SNN Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| num_steps | 10 | SNN simulation timesteps per forward pass |
| hidden_dim | 64 | Hidden layer neurons |
| beta | 0.999 | Membrane potential decay (0=no memory, 1=perfect) |
| spike_grad | sigmoid | Surrogate gradient for backprop |
| reset_mechanism | subtract | Membrane reset after spike |
| reset_delay | False | Delay reset by one timestep |

### Output Computation
1. Input is replicated across `num_steps` timesteps
2. Spikes are accumulated at each layer
3. Mean spike count is computed: `spikes.sum(dim=0) / num_steps`
4. Linear heads convert spike rates to continuous outputs

---

## Training (PPO)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Learning rate | 3e-4 (adaptive) |
| Num environments | 8192 |
| Horizon length | 256 steps |
| Minibatch size | 4096 |
| Mini epochs | 4 |
| Gamma | 0.99 |
| GAE lambda (tau) | 0.95 |
| Clip ratio | 0.2 |
| Entropy coef | 0.01 |
| Max epochs | 500 |

---

## File Structure

```
simple_hover_snn/
├── README.md                 # This file
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── task_config.py        # Observation/action dims, rewards, target
│   ├── env_config.py         # Environment bounds, physics settings
│   └── robot_config.py       # Spawn ranges, IMU config
├── task/
│   ├── __init__.py
│   └── simple_hover_snn_task.py  # Task logic, reward computation
├── networks/
│   ├── __init__.py
│   └── snn_netork.py         # SNN actor-critic network
└── training/
    ├── __init__.py
    ├── runner.py             # RL Games integration, registrations
    └── ppo_hover_snn.yaml    # PPO hyperparameters, SNN config
```

---

## Running

### Training
```bash
python simple_hover_snn/training/runner.py \
    --file=simple_hover_snn/training/ppo_hover_snn.yaml \
    --num_envs=4096 \
    --headless=True \
    --train
```

### Evaluation
```bash
python simple_hover_snn/training/runner.py \
    --file=simple_hover_snn/training/ppo_hover_snn.yaml \
    --checkpoint=<path_to_checkpoint> \
    --num_envs=64 \
    --headless=False \
    --play
```
