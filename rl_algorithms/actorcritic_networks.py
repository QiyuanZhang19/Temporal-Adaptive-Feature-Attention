import torch
import numpy as np
import torch.nn as nn
from rl_algorithms.utils import fanin_init, LayerNorm
from rl_algorithms.tanhdistributions import TanhNormal
LOG_SIG_MAX, LOG_SIG_MIN = 2, -20


class ActorAttention(nn.Module):
    def __init__(self, args):
        super(ActorAttention, self).__init__()
        self.args = args

        pos = torch.arange(0, self.args.obs_dims, dtype=torch.float, device=self.args.device).reshape(1, -1, 1)
        self.pos = (pos - float(self.args.obs_dims / 2.)) / np.sqrt(float(self.args.obs_dims))

        self.attention_k = nn.Sequential(nn.Linear(args.gru_hidden_size, args.qk_len), nn.ReLU(),
                                         nn.Linear(args.qk_len, args.qk_len))
        self.attention_q = nn.Sequential(nn.Linear(2, args.qk_len), nn.ReLU(),
                                         nn.Linear(args.qk_len, args.qk_len))
        self.attention_v = nn.Sequential(nn.Linear(2, args.v_len), nn.ReLU(),
                                         nn.Linear(args.v_len, args.v_len), nn.ReLU())
        self.v_qk_len = float(np.sqrt(args.v_len))
        # if self.args.layer_norm:
        #     self.before_gru_ln = LayerNorm(args.v_len)
        self.gru = nn.GRUCell(input_size=args.v_len + args.act_dims, hidden_size=args.gru_hidden_size)
        self._after_gru_mlp()
        self.actor_attention_weight = None      # track the temporal attention.
        if self.args.mask is not None:          # used to mask parts of features.
            self.mask = torch.tensor(self.args.mask, dtype=torch.float, device=self.args.device).reshape(1, -1)

    def _after_gru_mlp(self, b_init_value=0.1, init_w=3e-3):
        self.fcs = []
        self.layer_norms = []
        in_size = self.args.gru_hidden_size
        for i, next_size in enumerate(self.args.hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            fanin_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)

            if self.args.layer_norm:
                ln = LayerNorm(next_size)
                self.layer_norms.append(ln)

        self.fcs = nn.ModuleList(self.fcs)
        self.layer_norms = nn.ModuleList(self.layer_norms)
        self.last_fc = nn.Linear(in_size, self.args.act_dims)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        self.last_fc_log_std = nn.Linear(in_size, self.args.act_dims)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

    def _add_position(self, raw_x):
        """
        :param raw_x:        [b, obs_dims]
        :return:
        """
        assert len(raw_x.shape) == 2, raw_x.shape
        pos = self.pos.repeat(raw_x.shape[0], 1, 1)
        x_p = torch.cat([raw_x.unsqueeze(-1), pos], dim=-1)                     # [b, obs_dims, 2]
        return x_p

    def _attention(self, x_for_k, y_for_qv):
        """
        :param x_for_k:         the past history    [b, num_gru_hidden]
        :param y_for_qv:        tensor              [b, obs_dims, 2]
        :return:
        """

        k = self.attention_k(x_for_k.detach()).unsqueeze(1)                    # [b,        1, qk_len]
        q = self.attention_q(y_for_qv)                                         # [b, obs_dims, qk_len]
        v = self.attention_v(y_for_qv)                                         # [b, obs_dims, v_len]
        attention_aa = torch.sum(k * q, dim=-1) / self.v_qk_len                # [b, obs_dims]
        w_aa = torch.softmax(attention_aa, dim=-1)                             # (b, obs_dims)
        if self.args.mask is not None:
            w_aa = w_aa * self.mask
        aa = torch.sum(v * w_aa.unsqueeze(-1), dim=-2)                         # (b, v_len)
        self.actor_attention_weight = attention_aa.detach().cpu().numpy()
        return aa

    def forward(self, pre_gru_h, pre_act, cur_obs):
        cur_filtered_obs = self._attention(pre_gru_h, self._add_position(cur_obs))  # [b, num_basic]
        gru_input = torch.cat([pre_act, cur_filtered_obs], dim=-1)
        cur_gru_h = self.gru(gru_input, pre_gru_h)
        h = cur_gru_h
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.args.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = torch.relu(h)
        mean = self.last_fc(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        dist = TanhNormal(mean, std)
        return cur_gru_h, dist

    def get_init_h(self, ):
        h0 = np.zeros([self.args.gru_hidden_size])
        a0 = np.zeros([self.args.act_dims])
        return h0, a0


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.q1_gru = nn.GRUCell(input_size=args.obs_dims + args.act_dims, hidden_size=args.gru_hidden_size)
        self._q1_after_gru_mlp()

        self.q2_gru = nn.GRUCell(input_size=args.obs_dims + args.act_dims, hidden_size=args.gru_hidden_size)
        self._q2_after_gru_mlp()

    def _q1_after_gru_mlp(self, b_init_value=0.1, init_w=3e-3):
        self.q1_fcs = []
        self.q1_layer_norms = []
        in_size = self.args.gru_hidden_size + self.args.act_dims
        for i, next_size in enumerate(self.args.hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            fanin_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.q1_fcs.append(fc)

            if self.args.layer_norm:
                ln = LayerNorm(next_size)
                self.q1_layer_norms.append(ln)

        self.q1_fcs = nn.ModuleList(self.q1_fcs)
        self.q1_layer_norms = nn.ModuleList(self.q1_layer_norms)
        self.q1_last_fc = nn.Linear(in_size, 1)
        self.q1_last_fc.weight.data.uniform_(-init_w, init_w)
        self.q1_last_fc.bias.data.uniform_(-init_w, init_w)

    def _q2_after_gru_mlp(self, b_init_value=0.1, init_w=3e-3):
        self.q2_fcs = []
        self.q2_layer_norms = []
        in_size = self.args.gru_hidden_size + self.args.act_dims
        for i, next_size in enumerate(self.args.hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            fanin_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.q2_fcs.append(fc)

            if self.args.layer_norm:
                ln = LayerNorm(next_size)
                self.q2_layer_norms.append(ln)

        self.q2_fcs = nn.ModuleList(self.q2_fcs)
        self.q2_layer_norms = nn.ModuleList(self.q2_layer_norms)
        self.q2_last_fc = nn.Linear(in_size, 1)
        self.q2_last_fc.weight.data.uniform_(-init_w, init_w)
        self.q2_last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, pre_gru_h1, pre_gru_h2, pre_act, cur_obs, cur_act):
        """
        :param pre_gru_h1:      [b, s, gru_h]
        :param pre_gru_h2:      [b, s, gru_h]
        :param pre_act:         [b, s, act_dims]
        :param cur_obs:         [b, s, obs_dims]
        :param cur_act:         [b, s, obs_dims]
        :return:
        """
        gru_input = torch.cat([pre_act, cur_obs], dim=-1)
        cur_gru_h1 = self.q1_gru(gru_input, pre_gru_h1)                                          # [b, s, gru_h]
        mlp_inputs1 = torch.cat([cur_gru_h1, cur_act], dim=-1)
        h1 = mlp_inputs1
        for i, fc in enumerate(self.q1_fcs):
            h1 = fc(h1)
            if self.args.layer_norm and i < len(self.q1_fcs) - 1:
                h1 = self.q1_layer_norms[i](h1)
            h1 = torch.relu(h1)
        q1 = self.q1_last_fc(h1)                                                                    # [b, s, act_dims]

        gru_input = torch.cat([pre_act, cur_obs], dim=-1)
        cur_gru_h2 = self.q2_gru(gru_input, pre_gru_h2)                                      # [b, s, gru_h]
        mlp_inputs2 = torch.cat([cur_gru_h2, cur_act], dim=-1)
        h2 = mlp_inputs2
        for i, fc in enumerate(self.q2_fcs):
            h2 = fc(h2)
            if self.args.layer_norm and i < len(self.q2_fcs) - 1:
                h2 = self.q2_layer_norms[i](h2)
            h2 = torch.relu(h2)
        q2 = self.q2_last_fc(h2)                                                          # [b, s, act_dims]
        return cur_gru_h1, cur_gru_h2, q1, q2

    def Q1(self, pre_gru_h, pre_act, cur_obs, cur_act):
        gru_input = torch.cat([pre_act, cur_obs], dim=-1)
        cur_gru_h1 = self.q1_gru(gru_input, pre_gru_h)  # [b, s, gru_h]
        mlp_inputs1 = torch.cat([cur_gru_h1, cur_act], dim=-1)
        h1 = mlp_inputs1
        for i, fc in enumerate(self.q1_fcs):
            h1 = fc(h1)
            if self.args.layer_norm and i < len(self.q1_fcs) - 1:
                h1 = self.q1_layer_norms[i](h1)
            h1 = torch.relu(h1)
        q1 = self.q1_last_fc(h1)
        return cur_gru_h1, q1

    def get_init_h(self, ):
        h0 = np.zeros([self.args.gru_hidden_size])
        return h0


if __name__ == '__main__':
    class Config(object):
        def __init__(self):
            self.obs_dims = 4
            self.act_dims = 2
            self.gru_hidden_size = 64
            self.layer_norm = False
            self.hidden_sizes = [20, 20]
            self.qk_len = 10
            self.v_len = 40
            self.device = torch.device('cpu')

    args = Config()
    pi = ActorAttention(args)
    pre_gru_h = torch.zeros((1, args.gru_hidden_size), dtype=torch.float)
    pre_act = torch.zeros((1, args.act_dims), dtype=torch.float)
    cur_obs = torch.zeros((1, args.obs_dims), dtype=torch.float)
    res = pi.forward(pre_gru_h, pre_act, cur_obs)
    q1 = Critic(args)
    res = q1.forward(pre_gru_h, pre_gru_h, pre_act, cur_obs, pre_act)
    print(res)






