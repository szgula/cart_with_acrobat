
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from env_model import CartAcrobat

from ddpg_agent import Agent, timeit
np.set_printoptions(precision=4)



num_agents = 64
env = CartAcrobat(m=(0.2, 0.2, 0.2), b=(0.2, 0.1, 0.1), rail_lims=(-200, 200), num_instances=num_agents)

agent = Agent(state_size=6, action_size=1, random_seed=2, num_agents=num_agents)



def ddpg(n_episodes=1000, max_t=250, print_every=100, dt=0.01, action_scalar=5, debug=False):
    scores_deque = deque(maxlen=print_every)
    if debug:
        var1 = deque(maxlen=max_t)
        var2 = deque(maxlen=max_t)
        var3 = deque(maxlen=max_t)
        var4 = deque(maxlen=max_t)
    scores = []
    last_mean = 0
    for i_episode in range(1, n_episodes+1):
        Q = np.zeros((env.num_of_instances, env.n_q))
        init_noise = 0.5
        Q[:, 1:3] = init_noise * np.random.sample((num_agents, 2)) - 0.5 * init_noise
        #Q[:, 4:] = init_noise * np.random.sample((num_agents, 2)) - 0.5 * init_noise
        #Q[:, 1:3] += np.deg2rad(180)
        env.update_state(Q)

        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action_scalar * action, dt)
            energy = env.energy()
            #reward_2 = np.sum((energy[1] + energy[0])[:, 1:] / np.array(env.get_max_pot_energy()), axis=1) / 2
            #reward_2 = reward_2.clip(-1, 1)
            reward = reward.transpose()[:, 1].reshape((env.num_of_instances, 1))
            if debug:
                var1.append(reward_2[0])
                var2.append(reward[0, 0])
                var3.append(env.state[0, 1])
                var4.append(env.state[0, 2])

            #reward = 3 * reward + reward_2.reshape((env.num_of_instances, 1))
            #reward /= 4
            #print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += np.mean(reward)
            #print(reward[1])
        scores_deque.append(score)
        scores.append(score)
        agent.noise.update_noise_scalar(scalar=0.995)
        print('\rEpisode {}\tAverage Score: {:.2f}, Score: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % print_every == 0:
            mean = np.mean(scores_deque)
            ##if mean > last_mean + 1:

                #init_noise += .05
            #elif mean < last_mean - 0.1:
            #    agent.noise.update_noise_scalar(scalar=1.05)
            #    init_noise *= 0.99
            #elif mean < last_mean - 15:
            #    agent.noise.update_noise_scalar(theta=1)
            #    init_noise *= 0.2
            last_mean = mean
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_3.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_3.pth')
            
    return scores


agent.actor_local.load_state_dict(torch.load('checkpoint_actor_3.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic_3.pth'))
agent.actor_target.load_state_dict(torch.load('checkpoint_actor_3.pth'))
agent.critic_target.load_state_dict(torch.load('checkpoint_critic_3.pth'))
scores = ddpg(max_t=250)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()




agent.actor_local.load_state_dict(torch.load('checkpoint_actor_3.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic_3.pth'))

state = env.reset()
for t in range(200):
    action = agent.act(state, add_noise=False)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break 

env.close()

