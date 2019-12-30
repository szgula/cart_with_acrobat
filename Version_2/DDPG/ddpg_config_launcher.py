import os

def exe_ddpg(counter, tau, lr, gamma, cap, bs, up_it, ex_noise):
    os.system(
    r'/Users/magdalena.zastawnik-gula/Desktop/AGH/Robotics/venv/bin/python DDPG.py' +
    f' --name "train_loop_gamma_1_{counter}"'
    f' --tau {tau}' +
    f' --learning_rate {lr}' +
    f' --gamma {gamma}' +
    f' --capacity {cap}' +
    f' --batch_size {bs}' +
    f' --update_iteration {up_it}' +
    f' --exploration_noise {ex_noise}' +
    f' --max_episode {4000}' +
    f' --print_log {200}' +
    f' --log_interval {1000}'
    )

def do_sth(args):
    import torch
    import gym
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_name = os.path.basename(__file__)
    env = gym.make(args.env_name).unwrapped
    env.tau = args.simulation_step

    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    min_Val = torch.tensor(1e-7).float().to(device)  # min value

    directory = './exp' + script_name + args.env_name + './' + args.name + r'/'
    from DDPG import run_ddpg
    res = run_ddpg(args, env, state_dim, action_dim, min_Val, directory, device)
    print('==============')
    print(f'RESULT:     {res}')
    print('==============')


class MyConf:
    def __init__(self):
        self.name = 'core_simple_reward_pow_4_balance'
        self.mode = 'train'
        self.env_name = "CartAcrobat-v0"
        self.tau = 0.01
        self.target_update_interval = 1
        self.test_iteration = 100
        self.learning_rate = 5e-5
        self.gamma = 0.99
        self.capacity = 50000
        self.batch_size = 64
        self.seed = False
        self.random_seed = 9527
        self.update_iteration = 10
        self.exploration_noise = 0.1
        self.sample_frequency = 256
        self.render = False
        self.log_interval = 50
        self.load = False
        self.render_interval = 1
        self.max_episode = 10000
        self.max_length_of_trajectory = 2000
        self.print_log = 5
        self.simulation_step = 0.01
        self.calc_av_every = 5
        self.fill_to_learn = 0.1

counter = 0

tau = [0.01] #[0.001, 0.005, 0.01, 0.02, 0.05]
lr = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-2]
gamma = [0.99] # [0.999, 0.99, 0.9]
cap = [100000] #[10000, 25000, 50000, 75000, 100000]
bs = [256] #[16, 32, 64, 128]
up_it = [10] #[2, 5, 10, 15, 20]
ex_noise = [0.1] #[0.1, 0.3, 0.6]


conf = MyConf()
for tau_i in tau:
    for lr_i in lr:
        for gamma_i in gamma:
            for cap_i in cap:
                for bs_i in bs:
                    for up_it_i in up_it:
                        for ex_noise_i in ex_noise:
                            conf.learning_rate = lr_i
                            conf.gamma = gamma_i
                            counter += 1
                            print(counter, lr_i)
                            do_sth(conf)


