import time
from arguments import parse_args

import sumo_rl


if __name__ == '__main__':
    args = parse_args().parse_args()

    env = sumo_rl.parallel_env(net_file='./xml/2x2.net.xml',
                               route_file='./xml/2x2.rou.xml',
                               use_gui=True,
                               sumo_warnings=True,
                               num_seconds=100000)

    for i in range(10):
        observations = env.reset()
        print(observations[0])
        print("当前episode：", i+1)
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print("observations:=================================\n", observations)
            # print("actions:=================================\n", actions)
            time.sleep(0.1)