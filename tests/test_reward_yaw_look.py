"""Unit tests for the two yaw/dead-zone reward terms (R1 rectified heading, R2 look).

Drives the real _reward_progress against a stub `self`, so the arithmetic under test is
the shipped arithmetic — no reimplementation to drift out of sync. Distances are held
equal across steps so r_progress contributes exactly zero and the terms under test are
the only thing moving.

The property R2 exists to restore is the last test here: r_bearing is YAW-INVARIANT
(n and v are both rotated by the same yaw-only robot_vehicle_orientation, see
base_multirotor.py:290), so before R2 no reward term could distinguish a drone facing
its direction of travel from one flying the same path sideways.
"""
import math
import types

import pytest
import torch

from task.attitude_navigation_task import NavigationWithObstaclesTask
from config.task_config.f450_attitude_navigation_task_config import task_config

EMA_KEYS = ["r_heading", "r_progress", "p_speed", "p_jerk", "p_action_mag", "r_look"]


def _stub(lambda_look=0.0, rectify=False):
    """A minimal `self` carrying only what _reward_progress reads."""
    params = dict(task_config.reward_parameters)
    params["lambda_look"] = lambda_look
    params["heading_rectify"] = rectify

    stub = types.SimpleNamespace()
    stub.task_config = types.SimpleNamespace(
        reward_parameters=params, v_max=task_config.v_max
    )
    stub.device = "cpu"
    stub._ema_alpha = 0.02
    stub._reward_comp_ema = {k: 0.0 for k in EMA_KEYS}
    stub._action_scale = torch.tensor(
        [1.0, math.pi / 4, math.pi / 4, task_config.max_yaw_rate]
    )
    return stub


def _reward(stub, n, v, dist=5.0):
    """Evaluate _reward_progress with zero actions and zero net progress."""
    num = n.shape[0]
    d = torch.full((num,), dist)
    zeros = torch.zeros(num, 4)

    stub._direction_and_distance_to_target = lambda: (n, None)
    stub.obs_dict = {"robot_vehicle_linvel": v}
    stub.prev_dist = d
    stub.prev_action = zeros
    stub._get_dist_to_target = lambda: d

    mask = torch.ones(num, dtype=torch.bool)
    return NavigationWithObstaclesTask._reward_progress(stub, mask, zeros)


# Three envs, all with the target dead ahead in the vehicle frame:
# [0] flies at it, [1] flies 90 deg across, [2] flies directly away from it.
N_AHEAD = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
V_TOWARD_ACROSS_AWAY = torch.tensor([[2.0, 0, 0], [0, 2.0, 0], [-2.0, 0, 0]])


def test_unrectified_heading_taxes_retreat():
    """The behaviour R1 exists to remove: -lambda_b for every step of a retreat."""
    r = _reward(_stub(), N_AHEAD, V_TOWARD_ACROSS_AWAY)
    lambda_b = task_config.reward_parameters["lambda_b"]
    assert r[2].item() == pytest.approx(-lambda_b, abs=1e-6)


def test_rectified_heading_zeroes_the_retreat_tax_but_not_the_approach():
    base = _reward(_stub(), N_AHEAD, V_TOWARD_ACROSS_AWAY)
    rect = _reward(_stub(rectify=True), N_AHEAD, V_TOWARD_ACROSS_AWAY)

    assert rect[2].item() == 0.0, "retreat must be unrewarded, not penalized"
    assert rect[0].item() == base[0].item(), "approach must be untouched"


def test_look_term_pays_only_for_facing_the_direction_of_travel():
    lam = 0.08
    delta = _reward(_stub(lambda_look=lam), N_AHEAD, V_TOWARD_ACROSS_AWAY) - _reward(
        _stub(), N_AHEAD, V_TOWARD_ACROSS_AWAY
    )

    assert delta[0].item() == pytest.approx(lam, abs=1e-6), "aligned heading -> +lambda_look"
    assert delta[1].item() == 0.0, "flying sideways -> nothing"
    assert delta[2].item() == 0.0, "flying backwards -> nothing, and no tax"


def test_look_term_is_inert_below_the_speed_gate():
    """Near hover v_hat is dominated by vel_noise_std; paying for it rewards noise."""
    slow = torch.full((3, 3), 0.0)
    slow[:, 0] = task_config.reward_parameters["look_min_speed"] - 0.1

    gated = _reward(_stub(lambda_look=0.08), N_AHEAD, slow)
    baseline = _reward(_stub(), N_AHEAD, slow)
    assert torch.allclose(gated, baseline)


def test_look_term_is_the_only_yaw_sensitive_reward():
    """Same speed, same bearing-to-target, different heading -> only R2 notices.

    Env A flies along its own nose; env B flies the same speed 90 deg off its nose.
    Under the stock reward both score identically on every yaw-carrying term, which is
    precisely why the policy has no reason to yaw.
    """
    n = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])  # target ahead / target off the nose
    v = torch.tensor([[2.0, 0, 0], [0, 2.0, 0]])  # travel ahead / travel off the nose

    # Bearing (velocity vs target) is identical in both: cos = 1. Only the heading
    # relative to travel differs.
    stock = _reward(_stub(), n, v)
    assert stock[0].item() == pytest.approx(stock[1].item(), abs=1e-6), (
        "stock reward is yaw-invariant — this is the gap R2 fills"
    )

    with_look = _reward(_stub(lambda_look=0.08), n, v)
    assert with_look[0].item() > with_look[1].item(), "R2 must make yaw matter"
