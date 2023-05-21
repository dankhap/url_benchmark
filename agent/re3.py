import copy

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.ddpg import DDPGAgent


class RE3(nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 rnd_rep_dim,
                 encoder,
                 aug,
                 obs_shape,
                 obs_type,
                 k: int = 5,
                 average_entropy: bool = False,
                 clip_val=5.):
        super().__init__()
        self.clip_val = clip_val
        self.aug = aug

        if obs_type == "pixels":
            self.normalize_obs = nn.BatchNorm2d(obs_shape[0], affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_shape[0], affine=False)

        self.predictor = nn.Sequential(encoder, nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim))
        self.target = nn.Sequential(copy.deepcopy(encoder),
                                    nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, rnd_rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.k = k
        self.average_entropy = average_entropy

        self.apply(utils.weight_init)


    def forward(self, obs):
        obs = self.aug(obs)
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error


class RE3Agent(DDPGAgent):
    def __init__(self, rnd_rep_dim, update_encoder, beta: float= 0.05,
                 kappa: float = 0.000025,
                 latent_dim: int =128,
                 k: int = 5,
                 average_entropy: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.rnd_scale = rnd_scale
        self.update_encoder = update_encoder
        self.beta = beta
        self.kappa = kappa
        self.latent_dim = latent_dim
        self.k = k
        self.average_entropy = average_entropy

        self.rnd = RE3(self.obs_dim, self.hidden_dim, rnd_rep_dim,
                       self.encoder, self.aug, self.obs_shape,
                       self.obs_type).to(self.device)
        self.intrinsic_reward_rms = utils.RMS(device=self.device)

        # optimizers
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.lr)

        self.rnd.train()


    def compute_irs(self, samples: Dict, step: int = 0) -> th.Tensor:
            """Compute the intrinsic rewards for current samples.

            Args:
                samples (Dict): The collected samples. A python dict like
                    {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                    actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                    rewards (n_steps, n_envs) <class 'th.Tensor'>,
                    next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
                step (int): The global training step.

            Returns:
                The intrinsic rewards.
            """
            # compute the weighting coefficient of timestep t
            beta_t = self._beta * np.power(1.0 - self._kappa, step)
            num_steps = samples["obs"].size()[0]
            num_envs = samples["obs"].size()[1]
            obs_tensor = samples["obs"].to(self._device)

            intrinsic_rewards = th.zeros(size=(num_steps, num_envs)).to(self._device)

            with th.no_grad():
                for i in range(num_envs):
                    src_feats = self.random_encoder(obs_tensor[:, i])
                    dist = th.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
                    if self.average_entropy:
                        for sub_k in range(self.k):
                            intrinsic_rewards[:, i] += th.log(th.kthvalue(dist, sub_k + 1, dim=1).values + 1.0)
                        intrinsic_rewards[:, i] /= self.k
                    else:
                        intrinsic_rewards[:, i] = th.log(th.kthvalue(dist, self.k + 1, dim=1).values + 1.0)

            return intrinsic_rewards * beta_t
    def update_rnd(self, obs, step):
        metrics = dict()

        prediction_error = self.rnd(obs)

        loss = prediction_error.mean()

        self.rnd_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['rnd_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, step):
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = self.rnd_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # update RND first
        if self.reward_free:
            # note: one difference is that the RND module is updated off policy
            metrics.update(self.update_rnd(obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

            metrics['pred_error_mean'] = self.intrinsic_reward_rms.M
            metrics['pred_error_std'] = torch.sqrt(self.intrinsic_reward_rms.S)

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
