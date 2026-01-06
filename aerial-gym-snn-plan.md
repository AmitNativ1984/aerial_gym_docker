# SNN-based PPO Training for Aerial Gym Simulator - Implementation Plan

## Overview

This plan details the implementation of a Spiking Neural Network (SNN) actor-critic architecture integrated with PPO (Proximal Policy Optimization) for training aerial robots in the Aerial Gym Simulator.

**Key Specifications:**
- **SNN Framework:** snnTorch
- **Spike Encoding:** Rate coding (Poisson spike generation)
- **Task:** Position control (13D observation space)
- **Temporal Processing:** Frame stacking (N=4 frames)
- **Neuron Model:** Leaky Integrate-and-Fire (LIF)

---

## Architecture Summary

### Data Flow Pipeline

```
Environment (13D obs)
    ↓
Frame Stacker (4 frames → 52D)
    ↓
Rate Encoder (52D → 10 timesteps of spikes)
    ↓
SNN Backbone [LIF-256, LIF-128, LIF-64]
    ↓
    ├─→ Actor Head (LIF-13) → Rate Decoder → mu (actions)
    └─→ Critic Head (LIF-1) → Membrane Decoder → value
    ↓
PPO Algorithm (rl_games)
```

### Key Components

1. **Rate Encoder:** Converts continuous observations to Poisson spike trains
2. **LIF Neurons:** Temporal integration with membrane potential dynamics
3. **Frame Stacker:** Buffers last 4 observations for temporal context
4. **Spike Decoders:** Convert spike trains to continuous outputs
5. **Surrogate Gradients:** Arctangent approximation for backpropagation

---

## Implementation Steps

### Step 1: Create SNN Network Module

**File:** `/workspace/aerial_gym_simulator/aerial_gym/rl_training/rl_games/snn_network.py`

**Components to implement:**

#### 1.1 RateEncoder Class
- Input: `(batch, obs_dim)` continuous observations
- Output: `(num_steps, batch, obs_dim)` binary spike trains
- Method: Poisson rate coding via `spikegen.rate()`
- Parameters:
  - `num_steps=10`: SNN simulation timesteps
  - `gain=1.0`: Spike frequency scaling
  - `normalize=False`: Rely on rl_games normalization

#### 1.2 SpikeDecoder Class
- Input: `(num_steps, batch, features)` spike trains + optional membrane potential
- Output: `(batch, features)` continuous values
- Methods:
  - `'rate'`: Average spike count (for actor)
  - `'membrane'`: Final membrane potential (for critic)
  - `'weighted_sum'`: Exponentially weighted spikes

#### 1.3 FrameStacker Class
- Buffers last N observations
- Resets on episode termination (`dones` flag)
- Outputs flattened concatenation: `(batch, obs_dim * N)`

#### 1.4 SNNActorCritic Network
- **Shared Backbone:**
  - Linear(52, 256) + LIF
  - Linear(256, 128) + LIF
  - Linear(128, 64) + LIF

- **Actor Head:**
  - Linear(64, 13) + LIF
  - Rate decoder → mu (mean actions)

- **Critic Head:**
  - Linear(64, 1) + LIF
  - Membrane decoder → value estimate

- **Action Noise:**
  - Learnable or fixed `log_std` parameter
  - `sigma = exp(log_std)`

- **Forward Pass Logic:**
  1. Frame stack observations
  2. Encode to spikes
  3. Simulate SNN for `num_steps` timesteps
  4. Decode to continuous outputs
  5. Return `(mu, sigma, value, None)`

#### 1.5 SNNNetworkBuilder
- Implements `load(params)` to extract config
- Implements `build(name, **kwargs)` to construct network
- Follows rl_games `NetworkBuilder` interface

---

### Step 2: Register SNN Network

**File:** `/workspace/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runner.py`

**Modification:**

Add after existing imports:
```python
from aerial_gym.rl_training.rl_games.snn_network import SNNNetworkBuilder
from rl_games.algos_torch import model_builder

# Register custom SNN network
model_builder.register_network('snn_actor_critic', SNNNetworkBuilder)
```

This single addition makes the network available to all config files.

---

### Step 3: Create SNN Training Configuration

**File:** `/workspace/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_snn.yaml`

**Key sections:**

```yaml
network:
  name: snn_actor_critic  # Use custom SNN network
  separate: False  # Shared backbone

  snn:  # SNN-specific parameters
    num_steps: 10  # SNN timesteps per RL step
    beta: 0.95  # Membrane decay constant
    threshold: 1.0  # Spike threshold
    spike_grad: 'atan'  # Surrogate gradient method
    encoding_gain: 1.0  # Rate coding scaling
    decode_actor: 'rate'  # Actor decoding method
    decode_critic: 'membrane'  # Critic decoding method
    num_frames: 4  # Frame stacking buffer size
    hidden_sizes: [256, 128, 64]  # Layer dimensions
    learnable_sigma: False  # Fixed action noise

config:
  normalize_input: True  # CRITICAL: Normalize before encoding
  learning_rate: 1e-4  # May need tuning (try 5e-5 if unstable)
  num_actors: 8192  # Parallel environments (reduce to 4096 for 8GB GPU)
  mixed_precision: True  # Enable FP16 for 2x speedup
  horizon_length: 32
  mini_epochs: 4
  gamma: 0.99
  tau: 0.95
```

**Important configuration notes:**
- `normalize_input=True`: Ensures observations in [0,1] range for rate encoding
- `num_actors`: Reduce to 2048-4096 for GPUs with <12GB VRAM
- `mixed_precision`: Significantly speeds up SNN simulation
- Learning rate may need reduction if training is unstable

---

### Step 4: Training Execution

**Command:**
```bash
# Inside Docker container
cd /workspace/aerial_gym_simulator

python aerial_gym/rl_training/rl_games/runner.py \
    --file=./aerial_gym/rl_training/rl_games/ppo_aerial_quad_snn.yaml \
    --num_envs=8192 \
    --headless=True
```

**Monitoring:**
- Watch TensorBoard for reward curves, losses
- Monitor FPS (expect 5-10k vs 50k for standard MLP)
- Check for NaN gradients (indicates instability)
- Verify spike rates are 10-50% per layer (healthy range)

---

## Critical Implementation Details

### Temporal Dynamics Management

**Within single RL step (forward pass):**
- Membrane potentials propagate across `num_steps=10` SNN timesteps
- BPTT (backpropagation through time) accumulates gradients
- Spike history maintained for decoding

**Between RL steps:**
- Membrane potentials **reset to zero** (detached from computation graph)
- Frame stacker buffer **persists** until episode ends
- Episode termination (`dones=True`) resets frame buffer to zeros

**Rationale:**
- Prevents unbounded membrane accumulation
- Matches biological neuron inter-spike reset behavior
- Frame persistence captures multi-step temporal patterns

### Gradient Flow

```
Loss (PPO objective)
  ↓
∂L/∂mu, ∂L/∂value
  ↓
Spike Decoder gradients
  ↓
Surrogate gradient through spikes: ∂spike/∂membrane ≈ 1/(π(1+(πU)²))
  ↓
LIF neuron backward pass
  ↓
Linear layer weight updates
  ↓
Frame stacker (no learnable params, no gradients)
  ↓
Rate encoder (no learnable params, no gradients)
```

snnTorch handles surrogate gradient application automatically via `spike_grad` parameter.

### Memory and Compute Considerations

**Memory Usage:**
- Standard MLP: ~2GB VRAM (8192 envs)
- SNN (10 steps, 8192 envs): ~12GB VRAM
- SNN (10 steps, 4096 envs): ~6GB VRAM ← **Recommended for 8GB GPUs**

**Computational Overhead:**
- Baseline MLP: ~50,000 FPS
- SNN (10 steps): ~5,000-8,000 FPS (6-10x slower)
- SNN (5 steps): ~15,000 FPS (3x slower)

**Optimizations:**
- Enable `mixed_precision: True` (2x speedup with FP16)
- Reduce `num_snn_steps` from 10 to 5 (2x speedup)
- Reduce `num_actors` if memory-constrained
- Ensure all operations stay on GPU (no CPU transfers)

---

## Troubleshooting Guide

### Issue: Training Instability (NaN gradients, exploding losses)

**Solutions:**
1. Reduce learning rate: `1e-4 → 5e-5 or 1e-5`
2. Increase membrane decay: `beta: 0.95 → 0.98`
3. Increase SNN timesteps: `num_steps: 10 → 20` (smoother gradients)
4. Try different surrogate: `spike_grad: 'atan' → 'fast_sigmoid'`
5. Verify `normalize_input: True` is enabled

### Issue: Spike Saturation (neurons fire constantly >90%)

**Solutions:**
1. Reduce encoding gain: `encoding_gain: 1.0 → 0.5`
2. Increase spike threshold: `threshold: 1.0 → 1.5`
3. Check input normalization is working correctly

### Issue: Spike Silence (neurons never fire <1%)

**Solutions:**
1. Increase encoding gain: `encoding_gain: 1.0 → 1.5`
2. Reduce spike threshold: `threshold: 1.0 → 0.8`
3. Verify observations are in [0,1] range

### Issue: CUDA Out of Memory

**Solutions:**
1. Reduce parallel environments: `num_actors: 8192 → 4096 → 2048`
2. Reduce SNN timesteps: `num_steps: 10 → 5`
3. Enable mixed precision (already recommended)
4. Use gradient checkpointing (trades speed for memory)

### Issue: Slow Training (Low FPS)

**Solutions:**
1. Enable `mixed_precision: True` if not already
2. Reduce `num_snn_steps: 10 → 5`
3. Reduce `num_actors` (more actors = more SNN simulations)
4. Profile with PyTorch profiler to identify bottlenecks

---

## Hyperparameter Tuning Recommendations

### Priority 1 - Most Impactful

**`num_snn_steps` (SNN temporal resolution):**
- Start: 10
- Range: 5-20
- Impact: Higher = smoother gradients but slower training
- Tuning: If unstable, increase to 20; if too slow, reduce to 5

**`learning_rate`:**
- Start: 1e-4
- Range: 1e-5 to 1e-4
- Impact: SNNs may need lower LR than standard MLPs
- Tuning: If loss explodes, reduce to 5e-5 or 1e-5

**`beta` (membrane decay):**
- Start: 0.95
- Range: 0.9-0.99
- Impact: Higher = longer temporal memory
- Tuning: If unstable, increase to 0.98; if not learning, try 0.9

### Priority 2 - Secondary Parameters

**`encoding_gain`:**
- Start: 1.0
- Range: 0.5-1.5
- Impact: Controls spike frequency
- Tuning: Adjust based on spike rate monitoring

**`num_frames` (frame stacking):**
- Start: 4
- Range: 2-8
- Impact: More frames = more temporal context but higher dimensionality
- Tuning: Standard value is 4, rarely needs changing

**`threshold`:**
- Start: 1.0
- Range: 0.8-1.5
- Impact: Usually kept at 1.0
- Tuning: Only adjust if spike saturation/silence occurs

### Priority 3 - Advanced Tuning

**`spike_grad` (surrogate gradient):**
- Options: 'atan', 'fast_sigmoid', 'sigmoid'
- Default: 'atan' (smooth, gradient-friendly)
- Tuning: Try 'fast_sigmoid' if 'atan' is unstable

**`decode_actor` / `decode_critic`:**
- Options: 'rate', 'membrane', 'weighted_sum'
- Recommended: 'rate' for actor, 'membrane' for critic
- Tuning: Rarely needs changing

**`learnable_sigma`:**
- Default: False (fixed action noise)
- Alternative: True (learnable exploration)
- Tuning: Enable if you want adaptive exploration

---

## Expected Outcomes

### Performance Targets

**Training Time:**
- Baseline MLP: ~2-4 hours (8192 envs, 1000 epochs)
- SNN Target: ~10-20 hours (5-10x slower due to SNN overhead)

**Final Reward:**
- Baseline MLP: [Depends on task definition]
- SNN Target: 90-110% of baseline performance

**Computational Efficiency:**
- FPS: 5,000-10,000 (vs 50,000 baseline)
- Memory: 6-12GB VRAM (vs 2GB baseline)

### Success Criteria

**Minimum Viable Product (MVP):**
- ✓ SNN network trains without crashes/NaNs
- ✓ Reward increases over time (learning occurs)
- ✓ Final performance ≥70% of baseline MLP

**Full Success:**
- ✓ Final performance ≥90% of baseline
- ✓ Training stable across 3+ random seeds
- ✓ Spike rates in healthy range (10-50%)
- ✓ Memory usage ≤12GB VRAM

**Stretch Goals:**
- ✓ Final performance >100% (SNNs outperform standard networks)
- ✓ Training time <15 hours
- ✓ Architecture portable to neuromorphic hardware (Loihi, SpiNNaker)

---

## Files to Create/Modify

### New Files to Create

1. **`/workspace/aerial_gym_simulator/aerial_gym/rl_training/rl_games/snn_network.py`**
   - Main implementation file (~400-500 lines)
   - Contains: RateEncoder, SpikeDecoder, FrameStacker, SNNActorCritic, SNNNetworkBuilder

2. **`/workspace/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_snn.yaml`**
   - SNN-specific training configuration (~100 lines)
   - Defines all SNN hyperparameters and PPO settings

### Files to Modify

1. **`/workspace/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runner.py`**
   - Add 2 lines: import and register SNN network
   - Location: After existing imports, before main execution

### Reference Files (Read-Only)

- `/workspace/aerial_gym_simulator/aerial_gym/config/task_config/position_setpoint_task_config.py` - Understand observation space
- `/workspace/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml` - Baseline config reference
- Example custom networks from rl_games (for architecture patterns)

---

## Testing Strategy

### Unit Tests (Pre-Training Validation)

```python
# Test 1: Rate Encoder
encoder = RateEncoder(num_steps=10, gain=1.0)
obs = torch.rand(256, 13)
spikes = encoder(obs)
assert spikes.shape == (10, 256, 13)  # Correct shape
assert spikes.dtype == torch.float32  # Correct dtype
assert 0 <= spikes.mean() <= 1  # Valid spike rates

# Test 2: Spike Decoder
decoder = SpikeDecoder('rate')
decoded = decoder(spikes)
assert decoded.shape == (256, 13)  # Correct shape
assert not torch.isnan(decoded).any()  # No NaNs

# Test 3: Frame Stacker
stacker = FrameStacker(num_frames=4, obs_dim=13)
stacker.reset(256, 'cuda')
stacked = stacker.update(obs, torch.zeros(256, dtype=torch.bool))
assert stacked.shape == (256, 52)  # 13 * 4 frames

# Test 4: Full Network Forward Pass
network = SNNActorCritic(params={'snn': {}}, actions_num=13, input_shape=13)
obs_dict = {'obs': torch.rand(256, 13)}
mu, sigma, value, states = network(obs_dict)
assert mu.shape == (256, 13)  # Actions
assert value.shape == (256, 1)  # Value
assert not torch.isnan(mu).any()  # No NaNs
assert not torch.isnan(value).any()  # No NaNs
```

### Integration Tests (During Training)

**First 100 iterations:**
- Monitor for NaN gradients → Indicates instability
- Check reward increases → Validates learning
- Profile FPS and memory → Measure overhead
- Log spike rates → Verify neuron health (10-50% target)

**After 500 iterations:**
- Compare to baseline MLP at same iteration count
- Verify policy improves consistently
- Check for gradient vanishing/explosion

**Training to convergence:**
- Run for 1000+ epochs
- Test trained policy in simulator
- Measure final success rate
- Compare energy efficiency (if applicable)

---

## Key Methods and Patterns

### SNN Actor-Critic Forward Pass Pattern

```python
def forward(self, obs_dict):
    obs = obs_dict['obs']  # (batch, 13)

    # 1. Frame stacking: 13 → 52
    stacked_obs = self.frame_stacker.update(obs, dones)

    # 2. Spike encoding: 52 → (T=10, batch, 52)
    spike_input = self.encoder(stacked_obs)

    # 3. SNN backbone simulation
    mem1 = mem2 = mem3 = None
    spike_features = []

    for t in range(num_snn_steps):
        # Layer 1: LIF dynamics
        cur1 = self.fc1(spike_input[t])
        spk1, mem1 = self.lif1(cur1, mem1)

        # Layer 2
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        # Layer 3
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)

        spike_features.append(spk3)

    spike_features = torch.stack(spike_features)  # (T, batch, 64)

    # 4. Actor/Critic heads (parallel processing)
    actor_spikes = self.process_actor_head(spike_features)  # (T, batch, 13)
    critic_spikes = self.process_critic_head(spike_features)  # (T, batch, 1)

    # 5. Decode spikes to continuous values
    mu = self.actor_decoder(actor_spikes)  # (batch, 13)
    value = self.critic_decoder(critic_spikes, mem_critic)  # (batch, 1)
    sigma = torch.exp(self.log_std)  # (13,)

    return mu, sigma, value, None
```

### Surrogate Gradient Application (Automatic via snnTorch)

```python
# Define LIF with surrogate gradient
self.lif = snn.Leaky(
    beta=0.95,
    threshold=1.0,
    spike_grad=surrogate.atan()  # Automatic gradient substitution
)

# Forward pass (normal)
spk, mem = self.lif(current, mem)

# Backward pass (automatic surrogate)
# snnTorch replaces: d_spike/d_membrane = Heaviside'(x)
# with smooth approximation: d_spike/d_membrane ≈ 1/(π(1+(πx)²))
```

### Frame Buffer Management with Episode Resets

```python
def update(self, obs, dones):
    # Shift buffer (FIFO queue)
    self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
    self.buffer[:, -1, :] = obs  # Add newest observation

    # Reset frames for terminated episodes
    if dones.any():
        self.buffer[dones] = 0.0  # Zero out terminated episodes

    # Flatten for network input
    return self.buffer.reshape(obs.shape[0], -1)
```

---

## Architecture Rationale

### Why Shared Backbone?

**Advantages:**
- Parameter efficiency (fewer weights to learn)
- Aligned feature representations between actor and critic
- Faster training with limited data
- Standard practice in PPO implementations

**Trade-off:**
- Less flexibility for independent actor/critic learning
- Acceptable for this task (position control is relatively simple)

### Why Different Decoders for Actor/Critic?

**Actor (Rate Decoding):**
- Averages spikes over time window
- Produces stable mean action estimates
- Symmetric with rate encoding (conceptually clean)

**Critic (Membrane Decoding):**
- Uses final membrane potential
- Captures full temporal integration
- Richer information than spike count alone

### Why Frame Stacking vs Recurrent SNN?

**Frame Stacking (Chosen):**
- Simpler implementation and debugging
- Explicit temporal context across RL steps
- Compatible with existing RL algorithms
- Proven effective in standard RL (Atari, continuous control)

**Recurrent SNN (Alternative):**
- More biologically plausible
- Implicit temporal memory in membrane dynamics
- Harder to train (gradient flow through long sequences)
- Requires careful state initialization and reset logic

**Decision:** Start with frame stacking; can add recurrence later if needed.

---

## Research References

This implementation is based on recent SNN-RL research:

1. **Population-Coded SNNs for RL** (Tang et al., 2020)
   - Demonstrates SNNs can match/exceed ANNs in continuous control
   - Population coding achieves state-of-the-art performance
   - Our design uses simpler rate coding for initial implementation

2. **SNNs for Robotic Control** (Nature, 2024)
   - LIF neurons effective for continuous control tasks
   - Surrogate gradients enable stable training
   - Frame stacking improves temporal credit assignment

3. **snnTorch Framework** (Eshraghian et al.)
   - Provides PyTorch-compatible SNN primitives
   - Automatic surrogate gradient handling
   - Proven in multiple RL benchmarks

4. **rl_games Integration Patterns**
   - Custom network registration via `model_builder`
   - NetworkBuilder interface for config loading
   - Compatibility with existing PPO implementation

---

## Next Steps After Implementation

1. **Validate baseline performance**
   - Train standard MLP on position control
   - Establish baseline reward curve and training time
   - Use as comparison metric for SNN

2. **Implement and test SNN network**
   - Create `snn_network.py` with all components
   - Run unit tests to verify shapes and logic
   - Test network instantiation via rl_games

3. **Initial training runs**
   - Start with conservative settings (num_steps=10, beta=0.95)
   - Monitor for instability or crashes
   - Log spike rates and membrane potentials

4. **Hyperparameter optimization**
   - Grid search: num_snn_steps=[5,10,20], beta=[0.9,0.95,0.98]
   - Learning rate sweep: [1e-5, 5e-5, 1e-4]
   - Profile performance vs accuracy trade-off

5. **Scaling and optimization**
   - Benchmark memory usage and FPS
   - Implement optimizations if needed (gradient checkpointing, etc.)
   - Test on navigation task with vision (81D observations)

6. **Documentation and analysis**
   - Document final hyperparameters and performance
   - Analyze spike patterns and temporal dynamics
   - Evaluate neuromorphic deployment potential

---

## Summary

This plan provides a complete roadmap for implementing SNN-based PPO training in Aerial Gym Simulator. The architecture is designed to integrate seamlessly with the existing rl_games framework while introducing spiking neural network dynamics for temporal processing and potential energy efficiency gains.

**Core Innovation:** Combining rate-coded spike encoding, LIF neuron dynamics, frame stacking, and surrogate gradient training to enable end-to-end learning of spiking policies for continuous control.

**Expected Timeline:** 10-15 days from implementation start to validated results.

**Risk Mitigation:** Conservative hyperparameters, extensive troubleshooting guide, and fallback options for common issues ensure smooth development path.
