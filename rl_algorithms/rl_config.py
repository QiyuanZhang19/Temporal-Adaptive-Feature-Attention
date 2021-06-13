import yaml
import envs
import torch
import time
import gym
import os


class RLConfig(object):
    def __init__(self, exp_name, gpu_id=None, seed=0, hyper_file_name=None):
        self.exp_name = f'{exp_name}/seed_{seed}'
        self.seed = seed
        self.mask = None
        # 1. env config
        self.env = gym.make(exp_name)
        self.obs_dims = self.env.observation_space.shape[0]
        self.act_dims = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.file_name = f"./models/{self.exp_name}"
        self.save_model = True
        # 2. SAC config
        self.learnable_temperature = True
        self.policy_freq = 2
        self.lr = 3e-4
        self.discount = 0.99
        self.tau = 0.005
        # 3. buffer config
        self.max_timesteps = int(1e6)
        self.max_buffer_size = int(1e6)
        self.max_episode_step = self.env.spec.max_episode_steps  # flag_horizon_trick

        with open(f'{hyper_file_name}.yml', 'r', encoding="utf-8") as f:
            params_dict = yaml.load(f, Loader=yaml.FullLoader)
        for key in params_dict:
            setattr(self, key, params_dict[key])

        if gpu_id is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f"cuda:{gpu_id}")

        if not os.path.exists(f"./results/{self.exp_name}"):
            os.makedirs(f"./results/{self.exp_name}")



