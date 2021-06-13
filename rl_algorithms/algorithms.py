import copy
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from rl_algorithms.actorcritic_networks import ActorAttention, Critic


class DetectAlgorithm(object):
    def __init__(self, args):
        self.args = args
        self.actor = ActorAttention(args).to(self.args.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=self.args.lr)

        self.critic = Critic(args).to(self.args.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=self.args.lr)
        if args.learnable_temperature:
            self.target_entropy = -np.prod(self.args.env.action_space.shape[0]).item()
            self.log_alpha = torch.tensor(1., requires_grad=True, device=self.args.device)
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.args.lr)
        self.total_itr = 0

    def train(self, replay_buffer):
        self.total_itr += 1
        returns = replay_buffer.random_bootstrap_sample()
        self.update_critic(returns)
        if self.total_itr % self.args.policy_freq == 0:
            self.update_actor_and_alpha(returns)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1. - self.args.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1. - self.args.tau) * target_param.data)

    def update_critic(self, returns):
        with torch.no_grad():
            cur_gru_ahs = returns['cur_gru_ahs'][:, 0]
            cur_gru_ch1s = cur_gru_ch2s = returns['cur_gru_chs'][:, 0]
            target_qs = []
            for t in range(self.args.seq_len):
                next_gru_ahs, dist = self.target_actor.forward(cur_gru_ahs, returns['cur_acts'][:, t],
                                                               returns['next_obs'][:, t])
                next_action, pre_tanh_value = dist.sample(True)
                log_prob = dist.log_prob(next_action, pre_tanh_value).sum(-1, keepdim=True)
                next_gru_ch1s, next_gru_ch2s, q1, q2 = self.target_critic(cur_gru_ch1s, cur_gru_ch2s,
                                                                          returns['cur_acts'][:, t],
                                                                          returns['next_obs'][:, t], next_action)
                target_v = torch.min(q1, q2) - self.log_alpha.exp().detach() * log_prob
                target_q = returns['reward'][:, t] + self.args.discount * returns['not_done'][:, t] * target_v
                target_qs.append(target_q)
                cur_gru_ahs = next_gru_ahs * returns['cur_masks'][:, t]
                cur_gru_ch1s = next_gru_ch1s * returns['cur_masks'][:, t]
                cur_gru_ch2s = next_gru_ch2s * returns['cur_masks'][:, t]

        pre_gru_ch1s = pre_gru_ch2s = returns['pre_gru_chs'][:, 0]
        q1s, q2s = [], []
        for t in range(self.args.seq_len):
            cur_gru_ch1s, cur_gru_ch2s, q1, q2 = self.critic(pre_gru_ch1s, pre_gru_ch2s, returns['pre_acts'][:, t],
                                                             returns['cur_obs'][:, t], returns['cur_acts'][:, t])
            pre_gru_ch1s = self._mask_tensor(cur_gru_ch1s, returns['cur_masks'][:, t])
            pre_gru_ch2s = self._mask_tensor(cur_gru_ch2s, returns['cur_masks'][:, t])
            q1s.append(q1)
            q2s.append(q2)
        critic_loss = F.mse_loss(torch.stack(q1s, dim=1).reshape(-1, 1),
                                 torch.stack(target_qs, dim=1).reshape(-1, 1)) + \
                      F.mse_loss(torch.stack(q2s, dim=1).reshape(-1, 1),
                                 torch.stack(target_qs, dim=1).reshape(-1, 1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, returns):
        pre_gru_ahs = returns['pre_gru_ahs'][:, 0]
        pre_gru_chs = returns['pre_gru_chs'][:, 0]
        actor_loss = []
        log_probs = []
        for t in range(self.args.seq_len):
            cur_gru_ahs, dist = self.actor.forward(pre_gru_ahs, returns['pre_acts'][:, t], returns['cur_obs'][:, t])
            cur_action, pre_tanh_value = dist.rsample(True)
            log_prob = dist.log_prob(cur_action, pre_tanh_value).sum(-1, keepdim=True)
            log_probs.append(log_prob)
            cur_gru_chs, q1 = self.critic.Q1(pre_gru_chs, returns['pre_acts'][:, t], returns['cur_obs'][:, t],
                                             cur_action)
            assert log_prob.shape == q1.shape
            actor_loss.append(self.log_alpha.exp() * log_prob - q1)
            pre_gru_ahs = self._mask_tensor(cur_gru_ahs, returns['cur_masks'][:, t])
            pre_gru_chs = self._mask_tensor(cur_gru_chs, returns['cur_masks'][:, t])
        actor_loss = torch.stack(actor_loss).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_alpha(torch.stack(log_probs, dim=1).reshape(-1, 1))

    def update_alpha(self, log_prob):
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def _mask_tensor(self, x, mask):
        """
        :param x:         [b, obs_dims]
        :param mask:      [b,        1]
        :return:
        """
        z = torch.empty(x.shape, device=self.args.device)
        mask_fixed = torch.nonzero(mask)[:, 0]
        mask_update = torch.nonzero(1 - mask)[:, 0]
        z[mask_fixed] = x[mask_fixed]
        z[mask_update] = x[mask_update].detach() * 0.
        return z

    def save(self, filename):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()},
                   f"{filename}/{self.total_itr}.pth")

    def load(self, filename, num_itr):
        self.actor.load_state_dict(torch.load(f"{filename}/{num_itr}.pth", map_location='cpu')['actor'])
        self.critic.load_state_dict(torch.load(f"{filename}/{num_itr}.pth", map_location='cpu')['critic'])

    def get_init_hidden(self):
        ah0, a0 = self.actor.get_init_h()
        ch0 = self.critic.get_init_h()
        return ah0, ch0, a0













