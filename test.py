import gymnasium as gym
import sumo_rl

env = gym.make('sumo-rl-v0',
               net_file='./xml/1.net.xml', 
               route_file='./xml/1.rou.xml',
               out_csv_name='./xml/1.csv',
               use_gui=True,
               num_seconds=100000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated