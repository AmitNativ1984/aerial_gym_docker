"""Per-channel saturation and input sensitivity of a trained policy.

Answers two questions that losses/bounds_loss cannot, because it is a batch mean of a
SQUARED hinge summed over all four channels (rl_games a2c_common.py:171-179):

  1. WHICH channel saturates, and how often. Yaw chatter and thrust chatter are very
     different failures on the aircraft.
  2. Is the policy still a CONTROLLER or has it become a RELAY? The #17 commit's damning
     evidence for of4veb49 was not the number 42.4, it was "95% of outputs clamped and a
     0.1-sigma input change moved the command by 0.0000".

Runs the real task so mu is evaluated on the real state distribution -- channels are
strongly correlated (direction_to_target is a unit vector, the VAE latents are a learned
code), so sampling independent Gaussians from the normalizer statistics would measure a
distribution the policy never sees.

Deployment uses the Gaussian MEAN (deploy/checkpoint.py:13, player.deterministic), so mu
is the right quantity and action_log_std is deliberately ignored.

usage: python measure_saturation.py <checkpoint.pth> [n_steps] [num_envs]
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import re
import sys

import torch

import config  # noqa: F401  -- registers env/robot/task
from aerial_gym.registry.task_registry import task_registry

CKPT = sys.argv[1]
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
NUM_ENVS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
# Curriculum level to measure AT. Without this the task starts at
# task_config.curriculum.min_level (attitude_navigation_task.py:239), which is 0 --
# an EMPTY WORLD. Saturation measured with no obstacles says nothing about the
# demand that obstacle avoidance actually places on the action space.
LEVEL = int(sys.argv[4]) if len(sys.argv) > 4 else 25
TASK = "f450_navigation_task"
CHANNELS = ["thrust", "roll", "pitch", "yaw_rate"]

_NORM_EPS, _NORM_CLAMP = 1e-5, 5.0


def build_actor(weights):
    """Rebuild ANNMLPActor from the state_dict. Mirrors deploy/checkpoint.py."""
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
if "running_mean_std.running_mean" not in w:
    raise SystemExit("checkpoint has no input normalizer; refusing to guess")
mean = w["running_mean_std.running_mean"].float().to(dev)
std = torch.sqrt(w["running_mean_std.running_var"].float().to(dev) + _NORM_EPS)
print(f"checkpoint : {CKPT}")
print(f"epoch      : {ckpt.get('epoch')}   obs_dim {mean.numel()}")


def norm(o):
    return torch.clamp((o - mean) / std, -_NORM_CLAMP, _NORM_CLAMP)


# Pin the curriculum BEFORE building the task. min == max is the same path
# --curriculum_level uses (runner.py:154-158); obstacle density keys off the fixed
# density_at_level reference, so level 25 is 0.0670 obstacles/m^3 as everywhere else.
from config.task_config import F450NavTaskConfig
F450NavTaskConfig.curriculum.min_level = LEVEL
F450NavTaskConfig.curriculum.max_level = LEVEL
print(f"measuring at curriculum level {LEVEL} "
      f"(density {F450NavTaskConfig.obstacle_density_max * min(LEVEL/max(F450NavTaskConfig.curriculum.density_at_level,1),1.0):.4f}/m^3)")

task = task_registry.make_task(TASK, num_envs=NUM_ENVS, headless=True, use_warp=True)
obs = task.reset()[0]["observations"]

mus, sens, sens_pre = [], [], []
with torch.no_grad():
    for t in range(N_STEPS):
        n = norm(obs)
        mu = actor(n)
        mus.append(mu.clone())

        # Sensitivity: does a 0.1-sigma input nudge move the COMMAND the vehicle gets?
        # Perturb in normalized space (so 0.1 == 0.1 sigma per channel by construction).
        # Post-clamp is what reaches the controller; pre-clamp is what the network
        # itself computed. [A2] These can diverge once mu saturates: post-clamp
        # sensitivity collapses toward 0 in the flat region even if the network is
        # still just as reactive to inputs pre-clamp -- so a post-clamp-only
        # sensitivity drop over training is confounded with saturation itself and
        # cannot alone be read as "the policy desensitized".
        pert = actor(n + 0.1 * torch.randn_like(n))
        sens_pre.append((pert - mu).abs())
        sens.append((pert.clamp(-1, 1) - mu.clamp(-1, 1)).abs())

        obs = task.step(mu.clamp(-1, 1))[0]["observations"]

mu = torch.cat(mus)          # [N*envs, 4]
ds = torch.cat(sens)
ds_pre = torch.cat(sens_pre)
a = mu.abs()
n_tot = mu.shape[0]

print(f"samples    : {n_tot}  ({N_STEPS} steps x {NUM_ENVS} envs)\n")
print(f"{'channel':>9} {'mean|mu|':>9} {'p50':>7} {'p95':>7} {'max':>7} "
      f"{'%|mu|>1':>9} {'%>1.1':>7} {'sens_post':>10} {'sens_pre':>10}")
for i, c in enumerate(CHANNELS):
    ai = a[:, i]
    print(f"{c:>9} {ai.mean():>9.3f} {ai.median():>7.3f} "
          f"{torch.quantile(ai, 0.95):>7.3f} {ai.max():>7.3f} "
          f"{100*(ai > 1.0).float().mean():>8.1f}% {100*(ai > 1.1).float().mean():>6.1f}% "
          f"{ds[:, i].mean():>10.5f} {ds_pre[:, i].mean():>10.5f}")

any_sat = (a > 1.0).any(dim=1).float().mean()
all_sat = (a > 1.0).all(dim=1).float().mean()
print(f"\nany channel clamped : {100*any_sat:.1f}% of steps")
print(f"all 4 clamped       : {100*all_sat:.1f}% of steps")
print(f"mean |mu| over all  : {a.mean():.3f}")
print(f"mean command move for a 0.1-sigma input nudge (post-clamp): {ds.mean():.5f}")
print(f"mean command move for a 0.1-sigma input nudge (pre-clamp) : {ds_pre.mean():.5f}")
print("  (of4veb49 measured 0.0000 here -- a relay. Nonzero means still a controller.)")
print("  [A2] compare this run's pre-clamp sensitivity against an early-checkpoint run:")
print("  if pre-clamp sensitivity ALSO falls, the policy genuinely desensitized to")
print("  inputs; if only post-clamp falls, the earlier ep_50->ep_1800 drop was a")
print("  clamping artifact, not evidence the policy became a relay.")
task.close()
