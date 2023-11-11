import argparse


def parse_args():
    parser = argparse.ArgumentParser(description = "Reinforcement Learning for sumo environments!")

    parser.add_argument("--net_file", type=str, default='./xml/singlecrossing.net.xml', help="net.xml")
    parser.add_argument("--route_file", type=str, default='./xml/singlecrossing.rou.xml', help="route.xml")
    parser.add_argument("--use_gui", type=bool, default=False, help="是否打开GUI")
    parser.add_argument("--num_seconds", type=int, default=3600, help="每次模拟时间（s）")
    parser.add_argument("--speed", type=int, default=0.01, help="时间步长（s）")

    parser.add_argument("--n_episodes", type=int, default=1, help="轮数")
    parser.add_argument("--episode_limit", type=int, default=200, help="每轮step限制")
    parser.add_argument('--alg', type=str, default='qmix', help='算法')
    parser.add_argument('--n_steps', type=int, default=1500000, help='总step')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--model_dir', type=str, default='./model', help='模型保存路径')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果路径')
    parser.add_argument('--evaluate_epoch', type=int, default=16, help='评估轮数')
    parser.add_argument('--log_interval', type=int, default=2500, help='log间隔')
    parser.add_argument('--save_interval', type=int, default=10000, help='保存间隔')
    parser.add_argument('--evaluate_interval', type=int, default=5000, help='评估间隔')
    
    parser.add_argument('--load_model', type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--reuse_network', type=bool, default=True, help='重用网络')
    parser.add_argument('--last_action', type=bool, default=True, help='上一个动作')
    parser.add_argument('--two_hyper_layers', type=bool, default=False, help='两层超网络')
    
    
    parser.add_argument('--device', type=str, default='cuda', help='device used to train model; cpu, cuda')
    parser.add_argument('--rnn_hidden_dim', type=int, default=256, help='rnn dim')
    parser.add_argument('--qmix_hidden_dim', type=int, default=256, help='qmix dim')
    parser.add_argument('--lr', type=float, default=3e-4, help='lr')
    parser.add_argument('--epsilon', type=float, default=1., help='')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='')
    parser.add_argument('--anneal_steps', type=int, default=15000, help='')
    parser.add_argument('--train_steps', type=int, default=7, help='训练次数')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--buffer_size', type=int, default=5000, help='')
    parser.add_argument('--target_update_interval', type=int, default=300, help='soft更新间隔')
    parser.add_argument('--seed', type=int, default=27, help='')
   

    return parser

