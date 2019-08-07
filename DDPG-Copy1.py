#!/usr/bin/env python
# coding: utf-8

# # Deep Deterministic Policy Gradients (DDPG)
# ---
# In this notebook, we train DDPG with OpenAI Gym's Pendulum-v0 environment.
# 
# ### 1. Import the Necessary Packages

# In[1]:



import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent, timeit
np.set_printoptions(precision=4)


# In[12]:


class CartAcrobat:

    def __init__(self, l1=1, l2=1, m=(0.3, 0.2, 0.2), b=(0.2, 0.01, 0.01), g=9.81, rail_lims=(-3, 3), force_lims=(-20, 20), num_instances=1):
        self.g = np.float64(9.81)
        self.l1 = np.float64(l1)
        self.l2 = np.float64(l2)
        self.m = np.array(m, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.g = np.float64(g)
        self.rail_lims = np.array(rail_lims, dtype=np.float64)
        self.force_lims = np.array(force_lims, dtype=np.float64)

        # System dimensions
        self.n_u = 1  # inputs
        self.n_d = 3  # degrees of freedom
        self.n_q = 2 * self.n_d  # states

        self.init_state = [None]
        self.state = [None]
        self.time_elapsed = np.zeros(num_instances)
        self.num_of_instances = num_instances
        self.max_pot_1 = self.m[1] * self.l1
        self.max_pot_2 = self.m[2] * (self.l1 + self.l2)

    def update_state(self, q):
        # TODO: check if size of q coresponds to number of instances
        self.state = np.copy(q)
        self.init_state = np.copy(q)

    def get_position_to_plot(self):
        q = self.state[0]
        x = np.cumsum([q[0],
                       self.l1 * np.sin(q[1]),
                       self.l2 * np.sin(q[2])])
        y = np.cumsum([0,
                       self.l1 * np.cos(q[1]),
                       self.l2 * np.cos(q[2])])
        return (x, y)

    def energy(self):
        Ek = np.array([  0.5 * self.m[0] * (self.state[:, 3] ** 2),
                self.m[1] * ((self.state[:, 3] + self.l1 * self.state[:, 4] * np.cos(self.state[:, 1])) ** 2 +
                             (0 - self.l1 * self.state[:, 4] * np.sin(self.state[:, 1])) ** 2),
                self.m[1] * ((self.state[:, 3] + self.l1 * self.state[:, 4] * np.cos(self.state[:, 1]) + self.l2 * self.state[:, 5] * np.cos(self.state[:, 2])) ** 2 +
                             (0 - self.l1 * self.state[:, 4] * np.sin(self.state[:, 1]) - self.l2 * self.state[:, 5] * np.sin(self.state[:, 2])) ** 2)]).transpose()
        Ep = np.array([np.zeros((self.num_of_instances,)),
              self.m[1] * self.l1 * np.cos(self.state[:, 1]),
              self.m[1] * (self.l1 * np.cos(self.state[:, 1]) + self.l2 * np.cos(self.state[:, 2]))]).transpose()

        return Ek, Ep

    def get_max_pot_energy(self):
        return self.max_pot_1, self.max_pot_2
    
    def get_norm_hight(self):
        q = self.state
        h1 = np.cos(q[:, 1])
        h2 = (self.l1 * h1 + self.l2* np.cos(q[:, 2])) / (self.l1 + self.l2)
        return h1, h2

    def dstate_dt(self, u, dt=0.05):
        q = self.state
        theta1 = q[:, 1]
        theta2 = q[:, 2]
        dp = q[:, 3]
        dtheta1 = q[:, 4]
        dtheta2 = q[:, 5]

        b = [min(my_iter, 1 / dt - 1e-10) for my_iter in self.b]

        M = np.zeros((self.num_of_instances, 3,3))
        M[:, 0, 0] = np.sum(self.m)
        M[:, 0, 1] = self.l1 * (self.m[1] + self.m[2]) * np.cos(theta1)
        M[:, 0, 2] = self.m[2] * self.l2 * np.cos(theta2)

        M[:, 1, 0] = M[:, 0, 1]
        M[:, 1, 1] = (self.l1 ** 2) * (self.m[1] + self.m[2])
        M[:, 1, 2] = self.l1 * self.l2 * self.m[2] * np.cos(theta1 - theta2)

        M[:, 2, 0] = M[:, 0, 2]
        M[:, 2, 1] = M[:, 1, 2]
        M[:, 2, 2] = (self.l2 ** 2) * self.m[2]

        P1 = np.zeros((self.num_of_instances, 3,1))
        P1[:, 0, 0] = ((self.l1 * (self.m[1] + self.m[2]) * (dtheta1 ** 2) * np.sin(theta1))
                    + (self.m[2] * self.l2 * (dtheta2 ** 2) * np.sin(theta2)))
        P1[:, 1, 0] = ((-1) * self.l1 * self.l2 * self.m[2] * (dtheta2 ** 2) * np.sin(theta1 - theta2)
                    + self.g * (self.m[1] + self.m[2]) * self.l1 * np.sin(theta1))
        P1[:, 2, 0] = (self.l1 * self.l2 * self.m[2] * (dtheta1 ** 2) * np.sin(theta1 - theta2)
                    + self.g * self.l2 * self.m[2] * np.sin(theta2))

        P2 = b * self.state[:, self.n_d:]
        P2 = P2.reshape((self.num_of_instances,3, 1))

        P3 = np.zeros((self.num_of_instances, 3, 1))
        P3[:, 0, 0] = u.reshape(self.num_of_instances)

        P = P1 - P2 + P3

        d_y = np.matmul(np.linalg.inv(M), P)
        return(d_y.reshape((self.num_of_instances, 3)))


    #@timeit
    def step(self, u, dt, disturb=0.0):
        done = False
        q = self.state
        q = np.array(q, dtype=np.float64)

        # Enforce input saturation and add disturbance
        u_act = np.clip(np.float64(u), self.force_lims[0], self.force_lims[1]) + disturb

        # Get numpy views of pose and twist, and compute accel
        pose = np.copy(q[:, :self.n_d])
        twist = np.copy(q[:, self.n_d:])
        accel = self.dstate_dt(u, dt=dt)

        # Partial-Verlet integrate state and enforce rail constraints
        pose_next = pose + dt * twist + 0.5 * (dt ** 2)
        done = np.logical_or(self.state[:, 0] < self.rail_lims[0], self.state[:, 0] > self.rail_lims[1] )
        pose_next[:, 0] = np.clip(pose_next[:, 0], self.rail_lims[0], self.rail_lims[1])
        
        twist_next = twist + dt * accel

        # Return next state = [pose, twist]
        pose_next[:, 1:3] = np.mod(pose_next[:, 1:3] + np.pi, 2 * np.pi) - np.pi  # unwrap angles onto [-pi, pi]
        self.state = np.concatenate([pose_next, twist_next], axis=1)
        self.time_elapsed += dt
        reward = self.get_norm_hight()
        reward += done * -1000
        if np.any(done):
            self.reset(hard=False, done=done)
        done = done.reshape((self.num_of_instances, 1))
        return self.state, reward, done, ''
    
    def reset(self, hard=True, done=None):
        if hard:
            self.state = np.copy(self.init_state)
        else:
            self.state[done] = np.copy(self.init_state[done])
            self.time_elapsed[done] = 0
        return self.state


# ### 2. Instantiate the Environment and Agent

# In[13]:


num_agents = 64
env = CartAcrobat(m=(0.2, 0.2, 0.2), b=(0.2, 0.1, 0.1), rail_lims=(-200, 200), num_instances=num_agents)
Q = np.zeros((env.num_of_instances, env.n_q))
init_noise = 0.1
Q[:, 1:3] = init_noise * np.random.sample((num_agents,2)) - 0.5 * init_noise
#Q = np.random.sample((env.num_of_instances, env.n_q))
env.update_state(Q)
#env.seed(2)
agent = Agent(state_size=6, action_size=1, random_seed=2, num_agents=num_agents, freeze_agent=False)



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
        init_noise = 0.2
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
        agent.noise.update_noise_scalar(scalar=0.95)
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


#agent.actor_local.load_state_dict(torch.load('checkpoint_actor_3.pth'))
#agent.critic_local.load_state_dict(torch.load('checkpoint_critic_3.pth'))
#agent.actor_target.load_state_dict(torch.load('checkpoint_actor_3.pth'))
#agent.critic_target.load_state_dict(torch.load('checkpoint_critic_3.pth'))
scores = ddpg(max_t=250)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# ### 4. Watch a Smart Agent!

# In[25]:


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


# ### 6. Explore
# 
# In this exercise, we have provided a sample DDPG agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
# - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster than this benchmark implementation.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task!
# - Write your own DDPG implementation.  Use this code as reference only when needed -- try as much as you can to write your own algorithm from scratch.
# - You may also like to implement prioritized experience replay, to see if it speeds learning.  
# - The current implementation adds Ornsetein-Uhlenbeck noise to the action space.  However, it has [been shown](https://blog.openai.com/better-exploration-with-parameter-noise/) that adding noise to the parameters of the neural network policy can improve performance.  Make this change to the code, to verify it for yourself!
# - Write a blog post explaining the intuition behind the DDPG algorithm and demonstrating how to use it to solve an RL environment of your choosing.  
