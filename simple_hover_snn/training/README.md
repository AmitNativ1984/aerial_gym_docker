# Training Configuration

This directory contains training configurations for the simple hover task with two different network architectures.

## Network Options

### 1. Standard MLP Network (`ppo_hover_mlp.yaml`)

Standard feed-forward neural network matching the reference `position_setpoint_task` implementation:
- **Architecture**: [256, 128, 64] hidden layers with ELU activation
- **Action output**: Unbounded continuous values (no activation)
- **Learning rate**: 3e-4
- **Use case**: Baseline comparison, faster training, well-tested architecture

**Training command:**
```bash
python simple_hover_snn/training/runner.py \
    --file=simple_hover_snn/training/ppo_hover_mlp.yaml \
    --train \
    --num_envs=4096 \
    --headless=True
```

---

### 2. Spiking Neural Network (`ppo_hover_snn.yaml`)

Biologically-inspired spiking neural network using LIF (Leaky Integrate-and-Fire) neurons:
- **Architecture**: 3 LIF layers with 64 hidden units each
- **Spike dynamics**: 10 timesteps per forward pass, beta=0.999 (slow membrane decay)
- **Action output**: Linear rescaling of average spike rate [0,1] → [-1,1]
- **Learning rate**: 3e-3 (10x higher than MLP for SNN training)
- **Use case**: Energy-efficient inference, neuromorphic hardware deployment, biological plausibility

**Training command:**
```bash
python simple_hover_snn/training/runner.py \
    --file=simple_hover_snn/training/ppo_hover_snn.yaml \
    --train \
    --num_envs=4096 \
    --headless=True
```

---

## Reward Function

The task uses an **exponential position reward** with multiplicative bonuses, matching the reference `position_setpoint_task` architecture:

### Components:
1. **Position Reward** (Exponential curves)
   - Tight curve: `3.0 * exp(-8.0 * dist²)` - strong reward very close to goal
   - Broad curve: `2.0 * exp(-4.0 * dist²)` - attraction from distance
   - Provides multi-scale guidance (coarse + fine)

2. **Distance Reward** (Linear gradient)
   - `(2.0 - dist) / 4.0` - consistent pull toward goal even when far
   - Prevents local minima from exponential curves alone

3. **Attitude Bonus** (Multiplicative)
   - `0.2 / (0.1 + tiltage²)` - reward for being upright
   - **Multiplied by position reward** - matters more when close to goal

4. **Angular Velocity Bonus** (Multiplicative)
   - `3.0 / (1.0 + spinnage²)` - reward for being still
   - **Multiplied by position reward** - matters more when close to goal

5. **Action Smoothness Penalty** (CRITICAL for stability)
   - `0.05 * ||action - prev_action||` - penalizes jittery control
   - Reduces oscillations and instability

### Key Design:
- **All positive rewards** (except jitter penalty and crash)
- **Multiplicative bonuses** scale with position reward
- **No tunable cost coefficients** (proven hardcoded gains)
- Scaled for 1m flight zone: [-1, -1, -1] to [1, 1, 1]

## Key Differences

| Aspect | MLP | SNN |
|--------|-----|-----|
| **Hidden units** | [256, 128, 64] | [64, 64, 64] |
| **Activation** | ELU (continuous) | LIF neurons (spikes) |
| **Forward pass** | 1 step | 10 timesteps |
| **Learning rate** | 3e-4 | 3e-3 |
| **Entropy coef** | 0.01 | 0.01 |
| **Action output** | Unbounded | Linear rescale [0,1]→[-1,1] |
| **Training time** | Faster | ~10x slower (due to timesteps) |

---

## Everything Else is Identical

Both configurations use:
- **Same task**: `simple_hover_snn_task` (13D observations, 4D actions)
- **Same environment**: 1m spacing, no ground plane, 4096 parallel envs
- **Same reward function**: Exponential distance shaping + attitude bonuses
- **Same hyperparameters**: gamma=0.99, tau=0.95, horizon=256, minibatch=4096

---

## Testing / Inference

After training, test the policy with:

```bash
# Test MLP
python simple_hover_snn/training/runner.py \
    --file=simple_hover_snn/training/ppo_hover_mlp.yaml \
    --play \
    --checkpoint=runs/simple_hover_mlp/nn/simple_hover_mlp.pth

# Test SNN
python simple_hover_snn/training/runner.py \
    --file=simple_hover_snn/training/ppo_hover_snn.yaml \
    --play \
    --checkpoint=runs/simple_hover_snn/nn/simple_hover_snn.pth
```

---

## Tensorboard Monitoring

Monitor training progress:

```bash
tensorboard --logdir=runs/
```

Key metrics to watch:
- `rewards/frame`: Average episode return (should increase)
- `successes`: Number of successful hovers per step (should increase)
- `losses/policy_loss`: Policy gradient loss
- `losses/value_loss`: Value function loss
- `info/last_lr`: Current learning rate (adaptive)

---

## Expected Performance

Both networks should converge to hover at origin [0, 0, 0] within:
- **MLP**: 50-100 epochs (~30-60 minutes on GPU)
- **SNN**: 100-150 epochs (~2-4 hours on GPU)

Success criteria:
- Position error < 0.1m maintained for 2 seconds (40 steps)
- Reward > 10 (position + attitude bonuses)
