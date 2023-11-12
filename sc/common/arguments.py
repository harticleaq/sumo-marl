import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action 8')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven q_MAVEN qtran_haq_base
    parser.add_argument('--alg', type=str, default='quantilemix', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=1500000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--run_dir', type=str, default='./', help='result directory of the policy')
    parser.add_argument('--log_interval', type=int, default=20, help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')

    parser.add_argument('--td_type', type=str, default='td_lambda', help='how to compute global reward; td_lambda, max, qv')
    parser.add_argument('--q_total_type', type=str, default='individual', help='how to compute global reward; joint, individual')
    parser.add_argument('--qmix_type', type=str, default='mix2', help='how to compute mix q; mix, mix2')
    parser.add_argument('--device', type=str, default='cuda', help='device used to train model; cpu, cuda')
    parser.add_argument('--td_lambda', type=float, default=0.9, help='discount factor')
    
    args = parser.parse_args()
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 64
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 64
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 150000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 7

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 300

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 1

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001


    # QPLEX
    args.double_q = True
    args.mixing_embed_dim = 32
    args.hypernet_embed = 64
    args.adv_hypernet_layers= 1
    args.adv_hypernet_embed= 64

    args.num_kernel= 2
    args.is_minus_one= True
    args.is_adv_attention= True
    args.is_stop_gradient= True

    args.n_head= 2  # attention head number
    args.attend_reg_coef= 0.001  # attention regulation coefficient  # For MMM2 and 3s5z_vs_3s6z, it is 0.001
    args.state_bias= True  # the constant value c(s) in the paper
    args.mask_dead= False
    args.weighted_head= False  # weighted head Q-values, for MMM2 and 3s5z_vs_3s6z, it is True
    args.nonlinear= False  # non-linearity, for MMM2, it is True

    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10





    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args

