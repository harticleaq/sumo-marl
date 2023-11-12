import torch
import numpy
import os
from pathlib import Path
from runner import Runner
import time
from arguments import parse_args
import sumo_rl

# reward_fns = {
#         "diff-waiting-time": _diff_waiting_time_reward,
#         "average-speed": _average_speed_reward,
#         "queue": _queue_reward,
#         "pressure": _pressure_reward,
#     }

if __name__ == '__main__':
    args = parse_args().parse_args()
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / args.anneal_steps
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    numpy.random.seed(args.seed)
    env = sumo_rl.parallel_env(net_file=args.net_file,
                               route_file=args.route_file,
                               use_gui=args.use_gui,
                               sumo_warnings=True,
                               num_seconds=args.num_seconds)
    args.agent_id = env.possible_agents
    args.n_actions = env.action_space(env.possible_agents[0]).n
    args.obs_dim = max([env.observation_spaces[agent].shape[0] for agent in env.possible_agents])
    args.state_dim = sum([env.observation_spaces[agent].shape[0] for agent in env.possible_agents])
    args.state_shape = args.state_dim
    args.obs_shape = args.obs_dim
    args.n_agents = len(env.possible_agents)

    # 运行路径
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / args.alg / time.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_dir = str(run_dir)
    args.results_dir = str(Path(os.path.dirname(os.path.abspath(__file__)) + "/results")/ args.alg )

    runner = Runner(env, args)
    if not args.evaluate:
        runner.run(1)
    else:
        runner.evaluate()