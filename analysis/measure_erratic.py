"""Per-channel SATURATION and SMOOTHNESS of a trained policy, with and without obstacles.

The earlier script measured only the magnitude of mu. "Erratic" is a property of the
command over TIME, so this adds temporal metrics on the POST-CLAMP command -- what the
vehicle actually receives:

  |da|/step   mean absolute step-to-step change (the per-channel jerk that p_jerk sums)
  lag1 rho    autocorrelation at one step. A smooth command has rho near 1; white noise
              is near 0; an alternating (bang-bang) command goes NEGATIVE.
  flips/s     sign changes per second -- direct bang-bang counter
  swings/s    |da| > 1.0 in a single step, i.e. more than half the full command range
              traversed in 30 ms. These are the events that look like jitter.
  %at rail    fraction of steps with |command| >= 0.999

Transitions ACROSS an episode reset are excluded: a reset teleports the drone, so the
command discontinuity there is an artifact, not policy behaviour.

Deployment uses the Gaussian MEAN (deploy/checkpoint.py:13), so mu is the right quantity
and action_log_std is ignored.

usage: python measure_erratic.py <checkpoint.pth> <n_steps> <num_envs> <curriculum_level>

[A1] Noise ablation, via env vars (default: unchanged, i.e. noise ON, matching every
prior measurement in this file's history):
  A1_VEL_NOISE=0   disables F450NavTaskConfig.state_estimation_noise.enable
                   (velocity white noise into obs[4:7], config:108/114)
  A1_IMU_NOISE=0   disables GazeboImuConfig.enable_noise AND .enable_bias
                   (gyro noise+turn-on bias into obs[7:10], replacing ground truth)
Both default to "1" (noise on) so a plain invocation reproduces prior baselines exactly.
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import os
import re
import sys

import torch

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry

CKPT = sys.argv[1]
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
NUM_ENVS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
LEVEL = int(sys.argv[4]) if len(sys.argv) > 4 else 25
A1_VEL_NOISE = os.environ.get("A1_VEL_NOISE", "1") != "0"
A1_IMU_NOISE = os.environ.get("A1_IMU_NOISE", "1") != "0"
TASK = "f450_navigation_task"
CHANNELS = ["thrust", "roll", "pitch", "yaw_rate"]
DT = 0.03  # 800 steps x (sim dt 0.01 x 3 physics substeps) = 24 s  -> 33.3 Hz

_NORM_EPS, _NORM_CLAMP = 1e-5, 5.0


def build_actor(weights):
    idx = sorted(
        int(m.group(1))
        for k in weights
        if (m := re.fullmatch(r"a2c_network\.actor\.trunk\.(\d+)\.weight", k))
    )
    if not idx:
        raise SystemExit("not an ANNMLPActor checkpoint")
    layers = []
    for i in idx:
        w = weights[f"a2c_network.actor.trunk.{i}.weight"]
        b = weights[f"a2c_network.actor.trunk.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w); lin.bias.data.copy_(b)
        layers += [lin, torch.nn.ELU()]
    hw = weights["a2c_network.actor.action_head.weight"]
    hb = weights["a2c_network.actor.action_head.bias"]
    head = torch.nn.Linear(hw.shape[1], hw.shape[0])
    head.weight.data.copy_(hw); head.bias.data.copy_(hb)
    layers.append(head)
    return torch.nn.Sequential(*layers)


ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
w = ckpt["model"]
dev = "cuda:0"
actor = build_actor(w).to(dev).eval()
mean = w["running_mean_std.running_mean"].float().to(dev)
std = torch.sqrt(w["running_mean_std.running_var"].float().to(dev) + _NORM_EPS)


def norm(o):
    return torch.clamp((o - mean) / std, -_NORM_CLAMP, _NORM_CLAMP)


from config.task_config import F450NavTaskConfig
F450NavTaskConfig.curriculum.min_level = LEVEL
F450NavTaskConfig.curriculum.max_level = LEVEL
dens = F450NavTaskConfig.obstacle_density_max * min(
    LEVEL / max(F450NavTaskConfig.curriculum.density_at_level, 1), 1.0)

# [A1] Mutate the class objects BEFORE make_task builds anything from them. Both are
# referenced by other configs as the class itself (not an instance) -- f450_config.py:107
# sets `imu_config = GazeboImuConfig` -- so this mutation is visible wherever they're used.
F450NavTaskConfig.state_estimation_noise.enable = A1_VEL_NOISE
from config.sensor_config.gazebo_imu_config import GazeboImuConfig
GazeboImuConfig.enable_noise = A1_IMU_NOISE
GazeboImuConfig.enable_bias = A1_IMU_NOISE

print(f"checkpoint : {CKPT.split('/')[-1]}")
print(f"epoch {ckpt.get('epoch')}   level {LEVEL}   density {dens:.4f}/m^3")
print(f"[A1] vel_noise(state_estimation_noise.enable)={A1_VEL_NOISE}  "
      f"imu_noise(enable_noise+enable_bias)={A1_IMU_NOISE}")

task = task_registry.make_task(TASK, num_envs=NUM_ENVS, headless=True, use_warp=True)
obs = task.reset()[0]["observations"]

mus, cmds, dones, angvels, obsns = [], [], [], [], []
with torch.no_grad():
    for _ in range(N_STEPS):
        obs_n = norm(obs)
        mu = actor(obs_n)
        cmd = mu.clamp(-1, 1)
        mus.append(mu.clone()); cmds.append(cmd.clone())
        obsns.append(obs_n.clone())  # [A1-follow-up] normalized obs, for real dObs vs dmu
        o, _, term, trunc, _ = task.step(cmd)
        obs = o["observations"]
        dones.append((term | trunc).clone())
        # [A3] TRUE state, body frame -- task/attitude_navigation_task.py:693 uses this
        # same key as ground truth. Do NOT use obs[7:10]; that is the noisy simulated
        # gyro reading, not what the airframe actually does.
        angvels.append(task.obs_dict["robot_body_angvel"].clone())

MU = torch.stack(mus)      # [T, E, 4]
C = torch.stack(cmds)      # [T, E, 4]
D = torch.stack(dones)     # [T, E]
W = torch.stack(angvels)   # [T, E, 3] body angvel (rad/s), true state
ON = torch.stack(obsns)    # [T, E, obs_dim] normalized obs, as the network actually sees it

# valid transitions: t -> t+1 where the episode did not end at t
valid = ~D[:-1]                                   # [T-1, E]
dC = (C[1:] - C[:-1])                             # [T-1, E, 4]
vm = valid.unsqueeze(-1).expand_as(dC)
n_valid = valid.sum().item()

A = MU.abs()
print(f"samples {MU.shape[0]*MU.shape[1]}  valid transitions {n_valid}\n")
print(f"{'channel':>9}{'mean|mu|':>9}{'p50':>7}{'%|mu|>1':>9}"
      f"{'%at rail':>10}{'|da|/step':>11}{'lag1 rho':>10}{'flips/s':>9}{'swings/s':>10}")
for i, ch in enumerate(CHANNELS):
    a = A[:, :, i]
    c = C[:, :, i]
    d = dC[:, :, i][valid]
    # lag-1 autocorrelation over valid transitions only
    x, y = C[:-1, :, i][valid], C[1:, :, i][valid]
    xm, ym = x.mean(), y.mean()
    rho = (((x - xm) * (y - ym)).mean() /
           (x.std(unbiased=False) * y.std(unbiased=False) + 1e-9)).item()
    flips = ((torch.sign(x) != torch.sign(y)) & (x.abs() > 1e-3)).float().mean().item() / DT
    swings = (d.abs() > 1.0).float().mean().item() / DT
    rail = (c.abs() >= 0.999).float().mean().item()
    print(f"{ch:>9}{a.mean():>9.3f}{a.median():>7.3f}{100*(a>1).float().mean():>8.1f}%"
          f"{100*rail:>9.1f}%{d.abs().mean():>11.4f}{rho:>10.3f}{flips:>9.2f}{swings:>10.2f}")

dall = dC[vm].abs().mean().item()
print(f"\nmean |mu| all channels : {A.mean():.3f}")
print(f"mean |da| all channels : {dall:.4f} per step  ({dall/DT:.3f} per second)")

# [A3] Does command chatter reach the airframe, or does the plant filter it out?
# TRUE body rate (task.obs_dict["robot_body_angvel"]), not the noisy simulated gyro.
# Same reset-exclusion convention as the command metrics above: `valid` marks steps
# whose t -> t+1 transition does not cross an episode reset.
BODY_AXES = ["roll_rate", "pitch_rate", "yaw_rate"]
print(f"\nTRUE body rate (rad/s), post-plant, reset-excluded:")
print(f"{'axis':>11}{'RMS':>10}{'lag1 rho':>10}")
for i, ax in enumerate(BODY_AXES):
    w = W[:, :, i]
    x, y = w[:-1][valid], w[1:][valid]
    xm, ym = x.mean(), y.mean()
    rho = (((x - xm) * (y - ym)).mean() /
           (x.std(unbiased=False) * y.std(unbiased=False) + 1e-9)).item()
    rms = y.pow(2).mean().sqrt().item()
    print(f"{ax:>11}{rms:>10.4f}{rho:>10.3f}")
print("  If body-rate lag1 rho is high (near command rho) and RMS tracks the command's")
print("  |da|/step, the chatter reaches the airframe -- a physical bang-bang, not just")
print("  a command-space artifact. If body rates stay smooth while commands flip 5/s,")
print("  the plant is filtering it out and this is a deploy/actuator-wear concern, not")
print("  a flight-quality one -- target B2 (command-rate penalty), not B5 (body-rate")
print("  penalty), which would have nothing to grip.")

# [A1-follow-up] Does a SMALL real change in the observation produce a
# DISPROPORTIONATELY large action change? This is what CAPS's spatial-smoothness
# term penalizes (||mu(s) - mu(s+eps)||), measured here on the REAL trajectory this
# run actually produced (whatever A1_VEL_NOISE/A1_IMU_NOISE were set to) instead of a
# synthetic 0.1-sigma probe. A1 itself only asked whether REMOVING modeled noise
# reduces chatter (it doesn't); this asks whether the network is high-gain in
# general -- a property that would still matter even with all noise off, since the
# TRUE state changes a little every step regardless (motion, control response).
dON = ON[1:] - ON[:-1]                              # [T-1, E, obs_dim] normalized
dobs_norm = dON.norm(dim=-1)[valid]                 # ||d obs_norm||, per valid step
dmu_norm = (MU[1:] - MU[:-1]).norm(dim=-1)[valid]   # ||d mu||, same steps

corr = torch.corrcoef(torch.stack([dobs_norm, dmu_norm]))[0, 1].item()
print(f"\nSensitivity on the REAL trajectory (not a synthetic perturbation):")
print(f"  corr(||d obs_norm||, ||d mu||) across {dobs_norm.numel()} valid steps: {corr:.3f}")

n_bins = 5
order = torch.argsort(dobs_norm)
dobs_sorted, dmu_sorted = dobs_norm[order], dmu_norm[order]
n = dobs_sorted.numel()
print(f"\n  ||d obs_norm|| quintile -> mean ||d mu|| in that quintile:")
for i in range(n_bins):
    lo_idx, hi_idx = i * n // n_bins, (i + 1) * n // n_bins
    do, dm = dobs_sorted[lo_idx:hi_idx], dmu_sorted[lo_idx:hi_idx]
    gain = (dm.mean() / do.mean()).item()
    print(f"    q{i+1} obs-delta in [{do.min():.4f}, {do.max():.4f}]  "
          f"mean||d obs||={do.mean():.4f}  mean||d mu||={dm.mean():.4f}  gain={gain:.2f}")
print("  A roughly CONSTANT or RISING gain (mean||d mu||/mean||d obs||) across")
print("  quintiles means the network amplifies small real state changes about as much")
print("  as large ones -- high local gain independent of any noise model, which is")
print("  exactly the CAPS spatial term's target regardless of A1's verdict. A gain that")
print("  COLLAPSES in q1 would instead mean small real changes are handled")
print("  proportionally, and only the larger/rarer obs jumps drive the chatter.")
task.close()
