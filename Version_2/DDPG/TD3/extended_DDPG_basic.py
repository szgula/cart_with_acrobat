import torch
import torch.nn.functional as F
import torch.optim as optim
from TD3.replay_buffer import ReplayBuffer
from TD3.networks import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
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
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        return action

    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):

        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
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


def train():
    ######### Hyperparameters #########
    env_name = 'CartPoleContinuous-v0'  # "CartAcrobat-v0" #"BipedalWalker-v2" # "LunarLanderContinuous-v2"  #
    log_interval = 10  # print avg reward after interval
    random_seed = 0
    gamma = 0.99  # discount for future rewards
    batch_size = 100  # num of transitions sampled from replay buffer
    lr = 5e-1
    print(f'learning rate {lr}')
    exploration_noise = 0.1
    polyak = 0.995  # target policy update parameter (1-tau)
    policy_noise = 0.2  # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2  # delayed policy updates parameter
    max_episodes = 10000  # max num of episodes
    max_timesteps = 2000  # max timesteps in one episode
    directory = "./preTrained/{}".format(env_name)  # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    ###################################

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open("../log.txt", "w+")

    # training procedure:
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)

            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            avg_reward += reward
            ep_reward += reward

            # if episode is done then update policy:
            if done or t == (max_timesteps - 1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0

        # if avg reward > 300 then save and stop traning:
        if (avg_reward / log_interval) >= 475:
            print("########## Solved! ###########")
            name = filename + '_solved'
            policy.save(directory, name)
            log_f.close()
            break

        if episode > 500:
            policy.save(directory, filename)

        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0


if __name__ == '__main__':
    train()