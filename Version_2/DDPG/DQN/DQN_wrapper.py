import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from DQN.dqn_agent import Agent

env = gym.make('CartAcrobatDiscrete-v0')  # 'LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

agent = Agent(state_size=6, action_size=3, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

def dqn(n_episodes=100000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    max_ = 0
    max_mean = 0
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        save, save_mean = False, False
        for t in range(max_t):
            if i_episode % 100 == 0:
                env.render()
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        if score > max_:
            save = True
            max_ = score
        scores_window.append(score)  # save most recent score
        mean_score = np.mean(scores_window)
        if mean_score > max_mean:
            max_mean = np.mean(scores_window)
            save_mean = True
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        if save:
            torch.save(agent.qnetwork_local.state_dict(), r'DQN/results/checkpoint' + f"{max_:.2f}" + '.pth')
        if save_mean:
            torch.save(agent.qnetwork_local.state_dict(), r'DQN/results/mean_checkpoint' + f"{max_mean:.2f}" + '.pth')
        if mean_score >= 800.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            break
    return scores


scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


from time import sleep
while 1:
    state = env.reset()
    sum_reward = 0
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        sum_reward += reward
        if done:
            print(sum_reward)
            sleep(100)
            break