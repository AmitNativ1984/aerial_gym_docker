"""
A2C/PPO agent with a CAPS-style spatial-smoothness regularizer on the actor.

CAPS (Mysore et al., "Regularizing Action Policies for Smooth Control with
Reinforcement Learning", arXiv:2012.06644) splits smoothness into a TEMPORAL term
(penalize mu(s_t) vs mu(s_{t+1}) -- this repo already has that, it is p_jerk,
task/attitude_navigation_task.py:1344-1346) and a SPATIAL term:

    L_S = || mu(s) - mu(s + eps) ||^2,   eps ~ N(0, sigma^2)

which penalizes the actor for being sensitive to small input perturbations. This
is the training-time counterpart of the "0.1-sigma nudge" sensitivity probe
analysis/measure_saturation.py already uses to MEASURE the same quantity --
this class is that probe, made differentiable and added to the PPO loss instead
of only reported after the fact.

Why this is a candidate, and why it should not be the first thing tried: if the
roll/pitch/thrust erraticness measured in analysis/measure_erratic.py is driven
by the policy reacting to per-step observation noise (state_estimation_noise,
IMU gyro noise -- see config/task_config/f450_attitude_navigation_task_config.py
and config/sensor_config/gazebo_imu_config.py) rather than by the reward's
missing/underweighted action-rate term, then raising p_jerk's coefficient alone
will not fix it -- CAPS's spatial term is the literature-standard fix for exactly
that failure mode. But that diagnosis (noise vs. reward) is what the A1 ablation
in the smoothness/saturation plan settles, and it had not been run as of this
agent's introduction. Do not fund a long run with this enabled until A1 has run;
enabling it blind conflates two hypotheses the plan was explicitly structured to
keep separable.

Implementation notes:
  - Actor-only. Does NOT touch the critic.
  - The perturbed forward pass goes through the SAME live `self.model.a2c_network.
    actor` submodule that computes the real `mu` -- not a reconstructed copy (c.f.
    analysis/measure_saturation.py's build_actor(), which copies weights into a
    fresh nn.Sequential for post-hoc inference and is deliberately NOT reused here,
    because gradients from a copy would never reach the trainable parameters).
  - Normalization for the perturbed branch is computed directly from the current
    `running_mean_std` buffers with the same closed form analysis/measure_
    saturation.py and analysis/measure_erratic.py already use (mean, var, eps
    1e-5, clamp +/-5.0) -- NOT by calling running_mean_std(...) a second time,
    which would feed it synthetic (noise-perturbed) data and corrupt its EMA
    statistics with variance the real observation stream never has.
  - Not implemented for recurrent policies: perturbing a single timestep's
    observation independent of the hidden state it was produced with is not a
    well-posed sensitivity probe. Raises if caps.lambda_s > 0 on an RNN network.
"""
import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses

_NORM_EPS = 1e-5
_NORM_CLAMP = 5.0


class A2CCapsAgent(A2CAgent):
    """A2C/PPO agent that adds a CAPS spatial-smoothness loss to the actor.

    loss = ppo_loss + caps_lambda_s * || mu(s) - mu(s + eps) ||^2
    """

    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        caps_cfg = self.config.get('caps', {})
        self.caps_lambda_s = float(caps_cfg.get('lambda_s', 0.0))
        # In NORMALIZED obs units, same convention as measure_saturation.py's
        # "0.1-sigma nudge" (0.1 == 0.1 std per channel, by construction).
        self.caps_sigma = float(caps_cfg.get('sigma', 0.1))

        if self.caps_lambda_s > 0.0 and self.is_rnn:
            raise ValueError(
                "[a2c_caps] config.caps.lambda_s > 0 with a recurrent network is not "
                "supported (see module docstring) -- set it to 0.0 for RNN runs."
            )
        if self.caps_lambda_s > 0.0:
            print(f"[a2c_caps] spatial smoothness ON: lambda_s={self.caps_lambda_s:g} "
                  f"sigma={self.caps_sigma:g} (normalized-obs units)")
        else:
            print("[a2c_caps] spatial smoothness OFF (config.caps.lambda_s == 0) -- "
                  "this run is behaviorally identical to plain a2c_continuous.")

    def calc_gradients(self, input_dict):
        """PPO gradient step + CAPS spatial smoothness term.

        Re-implements A2CAgent.calc_gradients (rl_games a2c_continuous), kept
        structurally identical to the base -- and to agents/a2c_teacher_agent.py's
        own re-implementation -- so it tracks rl_games' loss/diagnostics exactly.
        The only addition is the caps_s_loss block.
        """
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            # --- CAPS spatial smoothness --------------------------------------------
            caps_s_loss = torch.zeros((), device=self.ppo_device)
            if self.caps_lambda_s > 0.0:
                with torch.no_grad():
                    mean = self.model.running_mean_std.running_mean.float()
                    var = self.model.running_mean_std.running_var.float()
                    std = torch.sqrt(var + _NORM_EPS)
                    obs_norm = torch.clamp((obs_batch.float() - mean) / std, -_NORM_CLAMP, _NORM_CLAMP)
                    obs_pert = torch.clamp(
                        obs_norm + self.caps_sigma * torch.randn_like(obs_norm),
                        -_NORM_CLAMP, _NORM_CLAMP,
                    )
                # Live actor submodule -- same parameters a_loss backprops into, so
                # this gradient joins the same backward()/optimizer step. Bypasses
                # the input normalizer module entirely: no second running_mean_std
                # update, no dependence on rl_games' internal norm_obs() naming.
                mu_pert, _ = self.model.a2c_network.actor(obs_pert.to(mu.dtype))
                caps_s_loss = (mu - mu_pert).pow(2).sum(dim=-1)
                (caps_s_loss_masked,), _ = torch_ext.apply_masks([caps_s_loss.unsqueeze(1)], rnn_masks)
                caps_s_loss = caps_s_loss_masked
                loss = loss + self.caps_lambda_s * caps_s_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(self,
        {
            'values': value_preds_batch,
            'returns': return_batch,
            'new_neglogp': action_log_probs,
            'old_neglogp': old_action_log_probs_batch,
            'masks': rnn_masks
        }, curr_e_clip, 0)

        if self.writer is not None and self.caps_lambda_s > 0.0:
            self.writer.add_scalar('caps/spatial_loss', float(caps_s_loss.detach()), self.frame)
            self.writer.add_scalar('caps/lambda_s', self.caps_lambda_s, self.frame)
            self.writer.add_scalar('caps/sigma', self.caps_sigma, self.frame)

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
