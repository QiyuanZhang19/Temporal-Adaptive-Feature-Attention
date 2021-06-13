import os
import copy
import torch
import pickle
import random
import numpy as np
from rl_algorithms.algorithms import DetectAlgorithm
from rl_algorithms.replay_buffer import ReplayBuffer


class Agent(object):
    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        self.solver = DetectAlgorithm(args)
        self.replay_buffer = ReplayBuffer(args)
        self.env_train = args.env
        self.env_eval = copy.deepcopy(args.env)
        self.attention_buffer = []
        self.t = 0

    def do_train(self):
        self.solver.train(self.replay_buffer)

    def get_action_train(self, pre_gru_ah, pre_gru_ch, pre_act, cur_obs):
        """
        :param pre_gru_ah:
        :param pre_gru_ch:
        :param pre_act:
        :param cur_obs:
        :return:
        """
        self.t += 1
        pre_gru_ah_tensor = torch.tensor(pre_gru_ah, dtype=torch.float, device=self.args.device).reshape(1, -1)
        pre_gru_ch_tensor = torch.tensor(pre_gru_ch, dtype=torch.float, device=self.args.device).reshape(1, -1)
        pre_act_tensor = torch.tensor(pre_act, dtype=torch.float, device=self.args.device).reshape(1, -1)
        cur_obs_tensor = torch.tensor(cur_obs, dtype=torch.float, device=self.args.device).reshape(1, -1)
        cur_gru_ah, dist = self.solver.actor(pre_gru_ah_tensor, pre_act_tensor, cur_obs_tensor)
        cur_act = dist.sample()

        cur_gru_ch, _ = self.solver.critic.Q1(pre_gru_ch_tensor, pre_act_tensor, cur_obs_tensor, cur_act)
        cur_act_np = cur_act.detach().cpu().numpy().squeeze(0)
        cur_gru_ah_np = cur_gru_ah.detach().cpu().numpy().squeeze(0)
        cur_gru_ch_np = cur_gru_ch.detach().cpu().numpy().squeeze(0)
        return cur_gru_ah_np, cur_gru_ch_np, cur_act_np

    def get_action_eval(self, pre_gru_ah, pre_act, cur_obs):
        """
        :param pre_gru_ah:
        :param pre_act:
        :param cur_obs:
        :return:
        """
        pre_gru_ah_tensor = torch.tensor(pre_gru_ah, dtype=torch.float, device=self.args.device).reshape(1, -1)
        pre_act_tensor = torch.tensor(pre_act, dtype=torch.float, device=self.args.device).reshape(1, -1)
        cur_obs_tensor = torch.tensor(cur_obs, dtype=torch.float, device=self.args.device).reshape(1, -1)
        cur_gru_ah, dist = self.solver.actor(pre_gru_ah_tensor, pre_act_tensor, cur_obs_tensor)
        cur_act = torch.tanh(dist.normal_mean)
        cur_act_np = cur_act.detach().cpu().numpy().squeeze(0)
        cur_gru_ah_np = cur_gru_ah.detach().cpu().numpy().squeeze(0)
        self.attention_buffer.append(self.solver.actor.actor_attention_weight)
        return cur_gru_ah_np, cur_act_np

    def eval_process(self, eval_episodes=10, cur_step=None):
        avg_reward = 0
        t = 0
        for itr in range(eval_episodes):
            self.attention_buffer = []
            cur_obs, done = self.env_eval.reset(), False
            pre_gru_ah, _, pre_act = self.solver.get_init_hidden()
            while not done:
                cur_gru_ah, cur_act = self.get_action_eval(pre_gru_ah, pre_act, cur_obs)
                next_obs, reward, done, _ = self.env_eval.step(cur_act)
                pre_gru_ah, pre_act = cur_gru_ah, cur_act
                cur_obs = next_obs
                avg_reward += reward
                t += 1
            if cur_step is not None:
                save_file = open(
                    f'./results/{self.args.exp_name}/policy_{cur_step}_attention_{itr}.pkl', 'wb')
                pickle.dump(np.asarray(self.attention_buffer), save_file)
        avg_reward /= eval_episodes
        t /= eval_episodes
        print("---------------------------------------"*4)
        print(f"Evaluation over {eval_episodes},  time step: {t: .3f},  episodes: {avg_reward:.3f}")
        print("---------------------------------------"*4)
        return avg_reward

    def load(self, file_name, num_itr):
        self.solver.load(file_name, num_itr)

    def training_process(self):
        self.env_train.seed(self.args.seed)
        self.env_eval.seed(self.args.seed + 100)
        evaluations = [self.eval_process(self.args.eval_episodes)]

        cur_obs, done = self.env_train.reset(), False
        pre_gru_ah, pre_gru_ch, pre_act = self.solver.get_init_hidden()
        episode_reward, episode_timesteps, episode_num = 0, 0, 0
        for t in range(int(self.args.max_timesteps)):
            episode_timesteps += 1
            cur_gru_ah, cur_gru_ch, cur_act = self.get_action_train(pre_gru_ah, pre_gru_ch, pre_act, cur_obs)
            next_obs, reward, done, _ = self.env_train.step(cur_act)

            # todo: store data in replay buffer
            self.replay_buffer.add_sample(pre_gru_ah, pre_gru_ch, pre_act, cur_obs, cur_act, cur_gru_ah,
                                          cur_gru_ch, next_obs, reward, done, episode_timesteps)
            pre_gru_ah, pre_gru_ch, pre_act = cur_gru_ah, cur_gru_ch, cur_act
            cur_obs = next_obs
            episode_reward += reward

            if t >= self.args.start_timesteps:
                self.do_train()

            if done:
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                cur_obs, done = self.env_train.reset(), False
                pre_gru_ah, pre_gru_ch, pre_act = self.solver.get_init_hidden()
                episode_reward, episode_timesteps, episode_num = 0, 0, 0

            if (t+1) % self.args.eval_freq == 0:
                evaluations.append(self.eval_process(self.args.eval_episodes))
                np.save(f"./results/{self.args.exp_name}/learning_process", evaluations)
                if self.args.save_model:
                    self.solver.save(f"./models/{self.args.exp_name}")








