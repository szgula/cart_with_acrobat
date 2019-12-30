import argparse
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

from collections import deque
import matplotlib.pyplot as plt
import json
import pickle

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''




class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 456)
        self.bn_2 = nn.BatchNorm1d(456)
        self.l3_1 = nn.Linear(456, 128)
        self.l4_1 = nn.Linear(128, 64)
        self.bn_4 = nn.BatchNorm1d(64 + state_dim)
        self.l5_1 = nn.Linear(64+state_dim, 32)
        self.l6_1 = nn.Linear(32, action_dim)


    def forward(self, input):
        x = F.relu(self.l1(input))
        x = self.l2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.l3_1(x)
        x = F.relu(x)
        x = F.relu(self.l4_1(x))
        x = torch.cat((x, input), 1)
        x = self.bn_4(x)
        x = F.relu(self.l5_1(x))
        x = torch.tanh(self.l6_1(x))
        return x



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.bn_0 = nn.BatchNorm1d(state_dim + action_dim) # TODO: normalize action on the input
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.bn_2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 300)
        self.l4 = nn.Linear(300, 50)
        self.bn_4 = nn.BatchNorm1d(50)
        self.l5 = nn.Linear(50, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = self.bn_0(x)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = self.bn_4(x)
        x = F.relu(x)
        x = self.l5(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, device, directory, args):
        self.directory = directory
        self.device = device
        self.args = args
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.args.learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.args.learning_rate)
        self.replay_buffer = Replay_buffer(self.args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def eval(self):
        self.actor.eval()
        #self.critic.eval()
        #self.actor_target.eval()
        #self.critic_target.eval()

    def train(self):
        self.actor.train()
        #self.critic.train()
        #self.actor_target.train()
        #self.critic_target.train()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(self.args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * self.args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self, max_=False):
        torch.save(self.actor.state_dict(), self.directory + 'actor.pth')
        torch.save(self.critic.state_dict(), self.directory + 'critic.pth')
        if max_:
            torch.save(self.actor.state_dict(), self.directory + 'actor_max.pth')
            torch.save(self.critic.state_dict(), self.directory + 'critic_max.pth')

    def load(self):
        self.actor.load_state_dict(torch.load(self.directory + 'actor_max.pth'))
        self.critic.load_state_dict(torch.load(self.directory + 'critic_max.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def get_max_state(max_state, state):
    for i, val in enumerate(state):
        max_state[i] = max(val, max_state[i])


def normalize_state(state, ranges):
    state[0] /= ranges[0]
    state[1] /= ranges[1]
    state[2] /= ranges[2]
    state[3] /= ranges[3]
    state[4] /= ranges[4]
    state[5] /= ranges[5]


def run_ddpg(args, env, state_dim, action_dim, min_Val, directory, device):
    max_state = [0]*6
    agent = DDPG(state_dim, action_dim, device, directory, args)
    scores_deque = deque(maxlen=50)
    all_av_scores = []
    max_ac_score = -10e9
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))

                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    #print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))

                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load()
        for i in range(args.max_episode):
            state = env.reset()
            #normalize_state(state, [env.x_threshold, 20, np.pi, 50, np.pi, 50])
            for t in count():
                agent.eval()
                action = agent.select_action(state)

                # issue 3 add noise to action
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

                next_state, reward, done, info = env.step(action)
                #normalize_state(next_state, [env.x_threshold, 20, np.pi, 50, np.pi, 50])
                #get_max_state(max_state, next_state)
                ep_r += reward
                if args.render and i >= args.render_interval : env.render()
                if not (done and t == 0):
                    agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                state = next_state
                if done or t >= args.max_length_of_trajectory:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    scores_deque.append(ep_r)
                    if i % args.calc_av_every == 0:
                        av_sc = np.mean(scores_deque)
                        all_av_scores.append(av_sc)
                    if i % args.print_log == 0:
                        # print("Ep_i \t{}, the ep_r is \t{:0.2f}, average \t{:0.2f}, the step is \t{}".format(i, ep_r, all_av_scores[-1], t))
                        print(f'({i}, {all_av_scores[-1]}),')
                    ep_r = 0
                    break

            if i % args.log_interval == 0:
                agent.save()
                #print('max_state', max_state)

            max_ = (av_sc > max_ac_score)
            if max_:
                agent.save(max_=max_)
                max_ac_score = max(av_sc, max_ac_score)

            if len(agent.replay_buffer.storage) >= (args.capacity * args.fill_to_learn) -1:
                agent.eval()
                agent.update()


        pickle_dict = {"av_score": all_av_scores}
        pickle.dump(pickle_dict, open(directory + "data.pickle", "wb"))
        save_config(args, directory)
        plt.plot(all_av_scores)
        plt.savefig(directory + "score.png")

    else:
        raise NameError("mode wrong!!!")

    return max_



def save_config(args, directory):
    config = {
        'name': args.name,
        'tau': args.tau,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'capacity': args.capacity,
        'batch_size': args.batch_size,
        'update_iteration': args.update_iteration,
        'exploration_noise': args.exploration_noise
    }
    with open(directory + 'config.json', 'w') as fp:
        json.dump(config, fp)

def get_config(get_conf_from_args=True):
    if get_conf_from_args:
        class MyConf:
            def __init__(self):
                self.name = 'core_simple_reward_pow_4_balance'
                self.mode = 'train'
                self.env_name = "CartAcrobat-v0"
                self.tau = 0.01
                self.target_update_interval = 1
                self.test_iteration = 100
                self.learning_rate = 5e-7
                self.gamma = 0.99
                self.capacity = 50000
                self.batch_size = 64
                self.seed = False
                self.random_seed = 9527
                self.update_iteration = 10
                self.exploration_noise = 0.3
                self.sample_frequency = 256
                self.render = False
                self.log_interval = 50
                self.load = False
                self.render_interval = 1
                self.max_episode = 50000
                self.max_length_of_trajectory = 2000
                self.print_log = 5
                self.simulation_step = 0.01
                self.calc_av_every = 5
                self.fill_to_learn = 0.1

        args = MyConf()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', default='core_simple_reward_pow_4_balance', type=str)

        parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
        parser.add_argument("--env_name", default="CartAcrobat-v0")
        parser.add_argument('--tau', default=0.01, type=float)  # target smoothing coefficient
        parser.add_argument('--target_update_interval', default=1, type=int)
        parser.add_argument('--test_iteration', default=100, type=int)

        parser.add_argument('--learning_rate', default=5e-5, type=float)
        parser.add_argument('--gamma', default=0.99, type=float)  # discounted factor
        parser.add_argument('--capacity', default=50000, type=int)  # replay buffer size
        parser.add_argument('--batch_size', default=64, type=int)  # mini batch size
        parser.add_argument('--seed', default=False, type=bool)
        parser.add_argument('--random_seed', default=9527, type=int)
        parser.add_argument('--update_iteration', default=10, type=int)
        parser.add_argument('--exploration_noise', default=0.3, type=float)  # 0.1
        # optional parameters

        parser.add_argument('--sample_frequency', default=256, type=int)
        parser.add_argument('--render', default=False, type=bool)  # show UI or not
        parser.add_argument('--log_interval', default=50, type=int)  #
        parser.add_argument('--load', default=False, type=bool)  # load model
        parser.add_argument('--render_interval', default=1, type=int)  # after render_interval, the env.render() will work
        parser.add_argument('--max_episode', default=50000, type=int)  # num of games
        parser.add_argument('--max_length_of_trajectory', default=2000, type=int)  # num of games
        parser.add_argument('--print_log', default=5, type=int)

        # my params
        parser.add_argument('--simulation_step', default=0.01, type=float)
        parser.add_argument('--calc_av_every', default=5, type=int)

        args = parser.parse_args()
    return args


def main():
    args = get_config()

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
    run_ddpg(args, env, state_dim, action_dim, min_Val, directory, device)


if __name__ == '__main__':
    main()

