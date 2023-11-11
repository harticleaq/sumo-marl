import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.args = args
        self.agents = agents
        self.agents_id = args.agents_id
        self.episode_limit = args.episode_limit
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if evaluate:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, u_onehot, terminate, padded = [], [], [], [], [], [], []
        u_onehot_next = []
        obs_dic = self.env.reset()[0]
        obs = [obs_dic[agent] for agent in self.agents_id]
        terminated = False
        step = 0
        episode_reward = 0  # cumulative rewards
        self.agents.policy.init_hidden(1)
        state = obs
        o.append(obs)
        s.append(state)
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        while not terminated and step < self.episode_limit:
            # time.sleep(0.1)
            actions, actions_onehot = [], []
            actions_dic = {}
            for agent_num in range(self.n_agents):
                agent_id = self.agents_id[agent_num]
                action = self.agents.choose_action(obs_dic[agent_id], agent_num, last_action[agent_num], epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                actions_dic[agent_id] = action
                last_action[agent_num] = action_onehot
            obs_dic, reward_dic, terminate_dic, _, _ = self.env.step(actions_dic)
            terminated = [terminate_dic[agent] for agent in self.agents_id].count(0) ==0
            obs = [obs_dic[agent] for agent in self.agents_id]
            reward = sum([reward_dic[agent] for agent in self.agents_id])
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            r.append([reward])
            terminate.append([int(terminated)])
            padded.append([0.])
            episode_reward += reward
            step += 1

            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

    
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        actions_onehot = []
        for agent_num in range(self.n_agents):
            agent_id = self.agents_id[agent_num]
            action = self.agents.choose_action(obs_dic[agent_id], agent_num, last_action[agent_num], epsilon, evaluate)
            action_onehot = np.zeros(self.args.n_actions)
            action_onehot[action] = 1
            actions.append(np.int(action))
            actions_onehot.append(action_onehot)
        u_onehot.append(actions_onehot)
        u_onehot_next = u_onehot[1:]
        u_onehot = u_onehot[:-1]
        

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            u_onehot_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
        
        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy(),
                       u_onehot_next=u_onehot_next.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        episode['s'] = episode['s'].squeeze(-2)
        episode['s_next'] = episode['s_next'].squeeze(-2)
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1:
            self.env.close()
        return episode, episode_reward, step


