import numpy as np
import torch


# Agent 
class Agent:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.n_agents = args.n_agents
        self.inputs_shape = args.obs_shape + args.n_agents + args.n_actions
        if args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        else:
            raise Exception("No such algorithm")
        self.args = args


    def choose_action(self, obs, agent_num, last_action, epsilon, evaluate=False):
        inputs = obs.copy()
        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        # transform the shape of inputs from (42,) to (1,42)

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.args.device)

        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        if np.random.uniform() < epsilon:
            action = np.random.choice(self.n_actions)  # action是一个整数
        else:
            action = int(torch.argmax(q_value))
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        train_infos = self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_interval == 0:
            self.policy.save_model(train_step)
        return train_infos
        









