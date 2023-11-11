import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agent
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agent(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.run_dir = self.args.run_dir
        self.log_dir = self.run_dir + '/logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)


    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if (time_steps // self.args.evaluate_interval) > evaluate_steps:
                episode_reward = self.evaluate()
                self.writter.add_scalars("evaluate_reward", {"evaluate_reward": episode_reward}, evaluate_steps)
                print("--ave reward is", episode_reward)
                self.episode_rewards.append(episode_reward)
                evaluate_steps += 1
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, reward, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                train_infos = self.agents.train(mini_batch, train_steps)
                train_steps += 1
                if train_steps % self.args.log_interval == 0:
                    for k, v in train_infos.items():
                        self.writter.add_scalars(k, {k: v[k]}, train_steps)


    def evaluate(self):
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward,  _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
        return episode_rewards / self.args.evaluate_epoch
