import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        self.size += 1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)

        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.bn_0 = nn.BatchNorm1d(state_dim)
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 456)
        self.bn_2 = nn.BatchNorm1d(456)
        self.l3_1 = nn.Linear(456, 128)
        self.l4_1 = nn.Linear(128, 64)
        self.bn_4 = nn.BatchNorm1d(64 + state_dim)
        self.l5_1 = nn.Linear(64+state_dim, 32)
        self.l6_1 = nn.Linear(32, action_dim)

    def forward(self, input):
        x = self.bn_0(input)
        x = self.l1(x)
        x = F.relu(x)
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


class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):

        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).reshape((batch_size, 1)).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip).reshape((batch_size, 1))
            next_action = (self.actor_target(next_state)[0] + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))

        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))

        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))

    def load(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

        self.critic_1.load_state_dict(
            torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(
            torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

        self.critic_2.load_state_dict(
            torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(
            torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

    def load_actor(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))


import torch
import gym
import numpy as np

def normalize_state(state, ranges):
    state[0] /= ranges[0]
    state[1] /= ranges[1]
    state[2] /= ranges[2]
    state[3] /= ranges[3]
    state[4] /= ranges[4]
    state[5] /= ranges[5]

def train():
    ######### Hyperparameters #########
    env_name = "CartAcrobat-v0"  # "BipedalWalker-v2" # "CartAcrobat-v0" #
    log_interval = 10  # print avg reward after interval
    random_seed = 0
    gamma = 0.99  # discount for future rewards
    batch_size = 256  # num of transitions sampled from replay buffer
    lr = 1e-1
    print(f'Learning Rate: {lr}')
    exploration_noise = 0.1
    polyak = 0.995  # target policy update parameter (1-tau)
    policy_noise = 0.2  # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2  # delayed policy updates parameter
    max_episodes = 2000  # max num of episodes
    max_timesteps = 500  # max timesteps in one episode
    save_timestemp = 500  #number of episodes between save action
    directory = "./preTrained/{}".format(env_name)  # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    load_pretrain = False
    normalize = False
    ###################################

    env = gym.make(env_name)
    env.num_env = 100
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)
    if load_pretrain:
        policy.load(directory, filename)
    replay_buffer = ReplayBuffer()

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open("log.txt", "w+")

    # training procedure:
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        if env_name == "CartAcrobat-v0" and normalize:
            normalize_state(state, [env.x_threshold, 20, np.pi, 50, np.pi, 50])
        for t in range(max_timesteps):
            # select action and add exploration noise:
            policy.actor.eval()
            action = policy.select_action(state)
            policy.actor.train()
            if env_name == "CartAcrobat-v0":
                size_ = env.num_env * env.action_space.shape[0]
            else:
                size_ = env.action_space.shape[0]
            action = action + np.random.normal(0, exploration_noise, size=size_)
            action = action.clip(env.action_space.low, env.action_space.high)

            # take action in env:
            next_state, reward, done, _ = env.step(action)
            #env.render()
            if env_name == "CartAcrobat-v0" and normalize:
                if abs(next_state[2]) > np.pi or abs(next_state[4]) > np.pi:
                    raise ValueError('angle out of range')
                normalize_state(next_state, [env.x_threshold, 20, np.pi, 50, np.pi, 50])
            for s, a, r, ns, d in zip(state, action, reward, next_state, done.astype('float')):
                replay_buffer.add((s, a, r, ns, d))
            state = next_state

            if env_name == "CartAcrobat-v0":
                reward = np.sum(reward) / env.num_env
            avg_reward += reward
            ep_reward += reward

            # if episode is done then update policy:
            if np.any(done) or t == (max_timesteps - 1):
                pass
                # policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                # break

        policy.update(replay_buffer, 100, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0

        # if avg reward > 300 then save and stop traning:
        if (avg_reward / log_interval) >= 300:
            print("########## Solved! ###########")
            name = filename + '_solved'
            policy.save(directory, name)
            log_f.close()
            break

        if episode != 0 and episode % save_timestemp == 0:
            policy.save(directory, filename)

        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = avg_reward / log_interval
            print(f"Episode: {episode}\tAverage Reward: {avg_reward:.2f}")
            avg_reward = 0


if __name__ == '__main__':
    train()