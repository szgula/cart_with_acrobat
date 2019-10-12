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

counter = 0

tau = [0.01] #[0.001, 0.005, 0.01, 0.02, 0.05]
lr = [1e-3] #[1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2]
gamma = [0.9, 0.99, 0.999] # [0.999, 0.99, 0.9]
cap = [50000] #[10000, 25000, 50000, 75000, 100000]
bs = [64] #[16, 32, 64, 128]
up_it = [10] #[2, 5, 10, 15, 20]
ex_noise = [0.1] #[0.1, 0.3, 0.6]


for tau_i in tau:
    for lr_i in lr:
        for gamma_i in gamma:
            for cap_i in cap:
                for bs_i in bs:
                    for up_it_i in up_it:
                        for ex_noise_i in ex_noise:
                            exe_ddpg(counter, tau_i, lr_i, gamma_i, cap_i, bs_i, up_it_i, ex_noise_i)
                            counter += 1
