import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h
from tqdm import tqdm


class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._dynamics = h.dynamics(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Qs = nn.ModuleList([h.q(cfg) for _ in range(cfg.num_q)])
        self._V = h.v(cfg)
        self.apply(h.orthogonal_init)
        for m in [self._reward, *self._Qs]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in self._Qs:
            h.set_requires_grad(m, enable)

    def track_v_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        if hasattr(self, '_V'):
            h.set_requires_grad(self._V, enable)

    def encode(self, obs):
        """Encodes an observation into its latent representation."""
        out = self._encoder(obs)
        if isinstance(obs, dict):
            # fusion
            out = torch.stack([v for k, v in out.items()]).mean(dim=0)
        return out

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def V(self, z):
        """Predict state value (V)."""
        return self._V(z)

    def Q(self, z, a, return_type):
        """Predict state-action value (Q)."""
        assert return_type in {'min', 'avg', 'all'}
        x = torch.cat([z, a], dim=-1)

        if return_type == 'all':
            return torch.stack(list(q(x) for q in self._Qs), dim=0)

        idxs = np.random.choice(self.cfg.num_q, 2, replace=False)
        Q1, Q2 = self._Qs[idxs[0]](x), self._Qs[idxs[1]](x)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2


class TDMPC():
    """Implementation of TD-MPC learning + inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = TOLD(cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.bc_optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.model.eval()
        self.model_target.eval()
        self.batch_size = cfg.batch_size

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict()}

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])
        self.model_target.load_state_dict(d['model_target'])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, step=None):
        """Take an action. Uses either MPC or the learned policy, depending on the self.cfg.mpc flag."""
        if isinstance(obs, dict):
            obs = {k: torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0) for k, o in obs.items()}
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.model.encode(obs)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, step=step)
        else:
            a = self.model.pi(z, self.cfg.min_std * (not eval_mode)).squeeze(0)
        return a.cpu()

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            if self.cfg.uncertainty_cost > 0:
                G -= discount * self.cfg.uncertainty_cost * self.model.Q(z, actions[t], return_type='all').std(dim=0)
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        pi = self.model.pi(z, self.cfg.min_std)
        G += discount * self.model.Q(z, pi, return_type='min')
        if self.cfg.uncertainty_cost > 0:
            G -= discount * self.cfg.uncertainty_cost * self.model.Q(z, pi, return_type='all').std(dim=0)
        return G

    @torch.no_grad()
    def plan(self, z, eval_mode=False, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        z: latent state.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        assert step is not None
        # Seed steps
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(_z, self.cfg.min_std)
                _z, _ = self.model.next(_z, pi_actions[t])

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples + num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std * torch.ones(horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device),
                                  -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (
                    score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, self.cfg.max_std)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return torch.clamp(a, -1, 1)

    def update_pi(self, zs, acts=None):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        self.model.track_v_grad(False)

        info = {}
        # Advantage Weighted Regression
        assert acts is not None
        vs = self.model.V(zs)
        qs = self.model_target.Q(zs, acts, return_type='min')
        adv = qs - vs
        exp_a = torch.exp(adv * self.cfg.A_scaling)
        exp_a = torch.clamp(exp_a, max=100.0)
        log_probs = h.gaussian_logprob(self.model.pi(zs) - acts, 0)
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = -((exp_a * log_probs).mean(dim=(1, 2)) * rho).mean()
        info['adv'] = adv[0]

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        self.model.track_v_grad(True)

        info['pi_loss'] = pi_loss.item()
        return pi_loss.item(), info

    @torch.no_grad()
    def _td_target(self, next_z, reward, mask):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_v = self.model.V(next_z)
        td_target = reward + self.cfg.discount * mask * next_v
        return td_target

    def update(self, replay_buffer, step, demo_buffer=None):
        """Main update function. Corresponds to one iteration of the model learning."""

        if demo_buffer is not None:
            # Update oversampling ratio
            self.demo_batch_size = int(
                h.linear_schedule(self.cfg.demo_schedule, step) * self.batch_size
            )
            replay_buffer.cfg.batch_size = self.batch_size - self.demo_batch_size
            demo_buffer.cfg.batch_size = self.demo_batch_size
        else:
            self.demo_batch_size = 0

        # Sample from interaction dataset
        obs, next_obses, action, reward, mask, done, idxs, weights = replay_buffer.sample()

        # Sample from demonstration dataset
        if self.demo_batch_size > 0:
            (demo_obs, demo_next_obses, demo_action, demo_reward, demo_mask, demo_done, demo_idxs, demo_weights
             ) = demo_buffer.sample()

            if isinstance(obs, dict):
                obs = {k: torch.cat([obs[k], demo_obs[k]]) for k in obs}
                next_obses = {k: torch.cat([next_obses[k], demo_next_obses[k]], dim=1) for k in next_obses}
            else:
                obs = torch.cat([obs, demo_obs])
                next_obses = torch.cat([next_obses, demo_next_obses], dim=1)
            action = torch.cat([action, demo_action], dim=1)
            reward = torch.cat([reward, demo_reward], dim=1)
            mask = torch.cat([mask, demo_mask], dim=1)
            done = torch.cat([done, demo_done], dim=1)
            idxs = torch.cat([idxs, demo_idxs])
            weights = torch.cat([weights, demo_weights])

        horizon = self.cfg.horizon
        loss_mask = torch.ones_like(mask, device=self.device)
        for t in range(1, horizon):
            loss_mask[t] = loss_mask[t - 1] * (~done[t - 1])

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(next_obses)
            z_targets = self.model_target.encode(next_obses)
            td_targets = self._td_target(next_z, reward, mask)

        # Latent rollout
        zs = torch.empty(horizon + 1, self.batch_size, self.cfg.latent_dim, device=self.device)
        reward_preds = torch.empty_like(reward, device=self.device)
        assert reward.shape[0] == horizon
        z = self.model.encode(obs)
        zs[0] = z
        value_info = {'Q': 0., 'V': 0.}
        for t in range(horizon):
            z, reward_pred = self.model.next(z, action[t])
            zs[t + 1] = z
            reward_preds[t] = reward_pred

        with torch.no_grad():
            v_target = self.model_target.Q(zs[:-1].detach(), action, return_type='min')

        # Predictions
        qs = self.model.Q(zs[:-1], action, return_type='all')
        value_info['Q'] = qs.mean().item()
        v = self.model.V(zs[:-1])
        value_info['V'] = v.mean().item()

        # Losses
        rho = torch.pow(self.cfg.rho, torch.arange(horizon, device=self.device)).view(-1, 1, 1)
        consistency_loss = (rho * torch.mean(h.mse(zs[1:], z_targets), dim=2, keepdim=True) * loss_mask).sum(dim=0)
        reward_loss = (rho * h.mse(reward_preds, reward) * loss_mask).sum(dim=0)
        q_value_loss, priority_loss = 0, 0
        for q in range(self.cfg.num_q):
            q_value_loss += (rho * h.mse(qs[q], td_targets) * loss_mask).sum(dim=0)
            priority_loss += (rho * h.l1(qs[q], td_targets) * loss_mask).sum(dim=0)

        self.expectile = h.linear_schedule(self.cfg.expectile, step)
        v_value_loss = (
            rho * h.l2_expectile(v_target - v, expectile=self.expectile) * loss_mask
        ).sum(dim=0)

        total_loss = (
                self.cfg.consistency_coef * consistency_loss +
                self.cfg.reward_coef * reward_loss +
                self.cfg.value_coef * q_value_loss +
                self.cfg.value_coef * v_value_loss
        )

        weighted_loss = (total_loss.squeeze(1) * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm,
                                                   error_if_nonfinite=False)
        self.optim.step()

        if self.cfg.per:
            # Update priorities
            priorities = priority_loss.clamp(max=1e4).detach()
            replay_buffer.update_priorities(
                idxs[: replay_buffer.cfg.batch_size], priorities[: replay_buffer.cfg.batch_size]
            )
            if self.demo_batch_size > 0:
                demo_buffer.update_priorities(demo_idxs, priorities[replay_buffer.cfg.batch_size:])

        # Update policy + target network
        _, pi_update_info = self.update_pi(zs[:-1].detach(), acts=action)

        if step % self.cfg.update_freq == 0:
            h.ema(self.model._encoder, self.model_target._encoder, self.cfg.tau)
            h.ema(self.model._Qs, self.model_target._Qs, self.cfg.tau)

        self.model.eval()
        metrics = {
            'consistency_loss': float(consistency_loss.mean().item()),
            'reward_loss': float(reward_loss.mean().item()),
            'Q_value_loss': float(q_value_loss.mean().item()),
            'V_value_loss': float(v_value_loss.mean().item()),
            'total_loss': float(total_loss.mean().item()),
            'weighted_loss': float(weighted_loss.mean().item()),
            'grad_norm': float(grad_norm),
        }
        for key in ["demo_batch_size", "expectile"]:
            if hasattr(self, key):
                metrics[key] = getattr(self, key)
        metrics.update(value_info)
        metrics.update(pi_update_info)

        return metrics
