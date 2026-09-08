"""Dump raw command traces (post-clamp, what the vehicle receives) to JSON for plotting.

usage: python dump_traces.py <checkpoint.pth> <level> <out.json> [n_steps] [n_envs]

[seed] optional override via env var F450_TRACE_SEED (default: task config's own
seed=42). Same seed across a batch of calls -> same reset draw order -> same
starting pose/target across checkpoints, which is what makes cross-checkpoint
trace comparison at a fixed target valid in the first place.
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import json
import os
import re
import sys

import torch

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry

CKPT, LEVEL, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
N_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
N_ENVS = int(sys.argv[5]) if len(sys.argv) > 5 else 16
_NORM_EPS, _NORM_CLAMP = 1e-5, 5.0


def build_actor(weights):
    idx = sorted(int(m.group(1)) for k in weights
                 if (m := re.fullmatch(r"a2c_network\.actor\.trunk\.(\d+)\.weight", k)))
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


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
w = ck["model"]
dev = "cuda:0"
actor = build_actor(w).to(dev).eval()
mean = w["running_mean_std.running_mean"].float().to(dev)
std = torch.sqrt(w["running_mean_std.running_var"].float().to(dev) + _NORM_EPS)

from config.task_config import F450NavTaskConfig
F450NavTaskConfig.curriculum.min_level = LEVEL
F450NavTaskConfig.curriculum.max_level = LEVEL
SEED = int(os.environ.get("F450_TRACE_SEED", F450NavTaskConfig.seed))
F450NavTaskConfig.seed = SEED
print(f"seed: {SEED}")

task = task_registry.make_task("f450_navigation_task", num_envs=N_ENVS,
                               headless=True, use_warp=True)
obs = task.reset()[0]["observations"]

cmds, mus, dones, speeds, bears = [], [], [], [], []
with torch.no_grad():
    for _ in range(N_STEPS):
        mu = actor(torch.clamp((obs - mean) / std, -_NORM_CLAMP, _NORM_CLAMP))
        cmd = mu.clamp(-1, 1)
        mus.append(mu.cpu()); cmds.append(cmd.cpu())
        o, _, term, trunc, _ = task.step(cmd)
        obs = o["observations"]
        dones.append((term | trunc).cpu())
        # observation_layout: slice(4, 7) is "linvel" -- vehicle-frame velocity.
        # The dict returned by step() has only "observations"; the raw tensors live on
        # task.obs_dict, so read the velocity out of the observation vector instead.
        speeds.append(torch.linalg.norm(obs[:, 4:7], dim=1).cpu())
        # observation_layout: slice(0, 3) is the UNIT direction to target in the VEHICLE
        # frame. The vehicle frame is yaw-only, so atan2(n_y, n_x) is exactly the bearing
        # error between where the drone points and where the target is: 0 = facing it.
        n_ = obs[:, 0:3]
        bears.append(torch.atan2(n_[:, 1], n_[:, 0]).cpu())

C = torch.stack(cmds).numpy()      # [T, E, 4]
M = torch.stack(mus).numpy()
D = torch.stack(dones).numpy()
S = torch.stack(speeds).numpy()

json.dump({
    "checkpoint": CKPT.split("/")[-1],
    "epoch": int(ck.get("epoch", -1)),
    "level": LEVEL,
    "dt": 0.03,
    "channels": ["thrust", "roll", "pitch", "yaw_rate"],
    "cmd": C.tolist(),      # post-clamp command, what the vehicle gets
    "mu": M.tolist(),       # pre-clamp network output
    "done": D.astype(int).tolist(),
    "speed": S.tolist(),
    "bearing_rad": torch.stack(bears).numpy().tolist(),
}, open(OUT, "w"))
print(f"wrote {OUT}: {C.shape[0]} steps x {C.shape[1]} envs")
task.close()
