import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.pre_gru_ahs = np.zeros((args.max_buffer_size, args.gru_hidden_size))
        self.cur_gru_ahs = np.zeros((args.max_buffer_size, args.gru_hidden_size))
        self.pre_gru_chs = np.zeros((args.max_buffer_size, args.gru_hidden_size))
        self.cur_gru_chs = np.zeros((args.max_buffer_size, args.gru_hidden_size))
        self.pre_act_nps = np.zeros((args.max_buffer_size, args.act_dims))
        self.cur_act_nps = np.zeros((args.max_buffer_size, args.act_dims))
        self.cur_obs_nps = np.zeros((args.max_buffer_size, args.obs_dims))
        self.next_obs_nps = np.zeros((args.max_buffer_size, args.obs_dims))
        self.reward_nps = np.zeros((args.max_buffer_size, 1))
        self.done_nps = np.zeros((args.max_buffer_size, 1))
        self.done_nps_ = np.zeros((args.max_buffer_size, 1))
        self.idx = 0
        self.size = 0

    def add_sample(self, pre_gru_ah, pre_gru_ch, pre_act, cur_obs, cur_act, cur_gru_ah, cur_gru_ch,
                   next_obs, reward, done, episode_timesteps):
        self.pre_gru_ahs[self.idx] = pre_gru_ah
        self.pre_gru_chs[self.idx] = pre_gru_ch
        self.cur_gru_ahs[self.idx] = cur_gru_ah
        self.cur_gru_chs[self.idx] = cur_gru_ch
        self.pre_act_nps[self.idx] = pre_act
        self.cur_obs_nps[self.idx] = cur_obs
        self.cur_act_nps[self.idx] = cur_act
        self.next_obs_nps[self.idx] = next_obs
        self.reward_nps[self.idx] = reward
        self.done_nps[self.idx] = done
        self.done_nps_[self.idx] = done
        if self.args.flag_horizon_trick and episode_timesteps == self.args.max_episode_step:
            self.done_nps[self.idx] = False

        self.idx = (self.idx + 1) % self.args.max_buffer_size

        if self.size < self.args.max_buffer_size:
            self.size += 1

    def random_bootstrap_sample(self):
        sample_idx_range = self.size - self.args.seq_len - 1
        batch_idx = np.random.choice(sample_idx_range, self.args.batch_size, replace=False)
        pre_gru_ahs, cur_gru_ahs, pre_gru_chs, cur_gru_chs = [], [], [], []
        pre_act_nps, cur_act_nps, cur_obs_nps, next_obs_nps = [], [], [], []
        reward_nps, not_done_nps, cur_mask_nps = [], [], []
        for idx in batch_idx:
            pre_gru_ahs.append(torch.tensor(self.pre_gru_ahs[idx:idx+self.args.seq_len],
                                            dtype=torch.float, device=self.args.device))
            pre_gru_chs.append(torch.tensor(self.pre_gru_chs[idx:idx + self.args.seq_len],
                                            dtype=torch.float, device=self.args.device))
            cur_gru_ahs.append(torch.tensor(self.cur_gru_ahs[idx:idx + self.args.seq_len],
                                            dtype=torch.float, device=self.args.device))
            cur_gru_chs.append(torch.tensor(self.cur_gru_chs[idx:idx + self.args.seq_len],
                                            dtype=torch.float, device=self.args.device))
            pre_act_nps.append(torch.tensor(self.pre_act_nps[idx:idx + self.args.seq_len],
                                            dtype=torch.float, device=self.args.device))

            cur_act_nps.append(torch.tensor(self.cur_act_nps[idx:idx + self.args.seq_len],
                                            dtype=torch.float, device=self.args.device))
            cur_obs_nps.append(torch.tensor(self.cur_obs_nps[idx:idx + self.args.seq_len],
                                            dtype=torch.float, device=self.args.device))
            next_obs_nps.append(torch.tensor(self.next_obs_nps[idx:idx + self.args.seq_len],
                                             dtype=torch.float, device=self.args.device))
            reward_nps.append(torch.tensor(self.reward_nps[idx:idx + self.args.seq_len],
                                           dtype=torch.float, device=self.args.device))
            not_done_nps.append(torch.tensor(1. - self.done_nps[idx:idx + self.args.seq_len],
                                             dtype=torch.float, device=self.args.device))
            cur_mask_nps.append(torch.tensor(1. - self.done_nps_[idx:idx + self.args.seq_len],
                                             dtype=torch.float, device=self.args.device))

        return {
            'pre_gru_ahs': torch.stack(pre_gru_ahs, dim=0),
            'pre_gru_chs': torch.stack(pre_gru_chs, dim=0),
            'cur_gru_ahs': torch.stack(cur_gru_ahs, dim=0),
            'cur_gru_chs': torch.stack(cur_gru_chs, dim=0),
            'pre_acts': torch.stack(pre_act_nps, dim=0),
            'cur_obs': torch.stack(cur_obs_nps, dim=0),
            'cur_acts': torch.stack(cur_act_nps, dim=0),
            'next_obs': torch.stack(next_obs_nps, dim=0),
            'reward': torch.stack(reward_nps, dim=0),
            'not_done': torch.stack(not_done_nps, dim=0),
            'cur_masks': torch.stack(cur_mask_nps, dim=0),
        }







