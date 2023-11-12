import torch
import sys
sys.path.append('/home/aqh/haq_pro/MA/StarCraft-master')
import numpy
from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
from pathlib import Path
import os
import time

if __name__ == '__main__':
    seed = list(range(1,9))

    # maps = ['MMM2']
    # maps = ['5m_vs_6m']
    # maps = ['2s_vs_1sc']
    # maps = ['3s5z']
    maps=['3s_vs_5z']
    # maps = ['3m']
    # maps = ['8m']
    
    types = [0, 1, 2]
    for i in range(5, 12):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        args.map = maps[0]
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.unit_dim = 4 + env.shield_bits_ally + env.unit_type_bits
        args.seed = 123
        
        t = types[i % 3]
        if t == 0:
            args.q_total_type = "individual"
        elif t == 1:
            args.q_total_type = "joint"
            args.td_type = "max"
        elif t == 2:
            args.q_total_type = "joint"
            args.td_type = "td_lambda"
        # 运行路径
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/StarCraft-master/results") /args.map / args.alg / time.strftime("%Y-%m-%d, %H:%M:%S")
        args.run_dir = str(run_dir)

        print(args.map)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        numpy.random.seed(args.seed)
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
