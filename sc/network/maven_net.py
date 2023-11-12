import torch
import torch.nn as nn
import torch.nn.functional as f


# output prob of z for an episode
class HierarchicalPolicy(nn.Module):
    def __init__(self, args):
        super(HierarchicalPolicy, self).__init__()
        self.fc_1 = nn.Linear(args.state_shape, 128)
        self.fc_2 = nn.Linear(128, args.noise_dim)

    def forward(self, state):
        x = f.relu(self.fc_1(state))
        q = self.fc_2(x)
        prob = f.softmax(q, dim=-1)
        return prob

class QtranQBase(nn.Module):
    def __init__(self, args):
        super(QtranQBase, self).__init__()
        self.args = args
        # action_encoding对输入的每个agent的hidden_state和动作进行编码，从而将所有agents的hidden_state和动作相加得到近似的联合hidden_state和动作
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                             nn.ReLU(),
                                             nn.Linear(ae_input, ae_input))

        # 编码求和之后输入state、所有agent的hidden_state和动作之和
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    # 因为所有时刻所有agent的hidden_states在之前已经计算好了，所以联合Q值可以一次计算所有transition的，不需要一条一条计算。
    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)  # 变回n_agents维度用于求和
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)

        inputs = torch.cat([state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1)
        q = self.q(inputs)
        q = -torch.exp(q)
        return q

class BootstrappedRNN(nn.Module):
    def __init__(self, input_shape, args):
        super(BootstrappedRNN, self).__init__()
        self.args = args

        self.fc = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.hyper_w = nn.Linear(args.noise_dim + args.n_agents, args.rnn_hidden_dim * args.n_actions)
        self.hyper_b = nn.Linear(args.noise_dim + args.n_agents, args.n_actions)

    def forward(self, obs, hidden_state, z):
        agent_id = obs[:, -self.args.n_agents:]
        hyper_input = torch.cat([z, agent_id], dim=-1)

        x = f.relu(self.fc(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        h = h.view(-1, 1, self.args.rnn_hidden_dim)

        hyper_w = self.hyper_w(hyper_input)
        hyper_b = self.hyper_b(hyper_input)
        hyper_w = hyper_w.view(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        hyper_b = hyper_b.view(-1, 1, self.args.n_actions)

        q = torch.bmm(h, hyper_w) + hyper_b
        q = q.view(-1, self.args.n_actions)
        return q, h


# variational distribution for MI Loss， output q(z|sigma(tau))
class VarDistribution(nn.Module):
    def __init__(self, args):
        super(VarDistribution, self).__init__()
        self.args = args

        self.GRU = nn.GRU(args.n_agents * args.n_actions + args.state_shape, 64)

        self.fc_1 = nn.Linear(64, 32)
        self.fc_2 = nn.Linear(32, args.noise_dim)

    def forward(self, inputs):  # q_value.
        # get sigma(q) by softmax
        _, h = self.GRU(inputs)  # (1, 1, 64)
        x = f.relu(self.fc_1(h.squeeze(0)))
        x = self.fc_2(x)
        output = f.softmax(x, dim=-1)
        return output
