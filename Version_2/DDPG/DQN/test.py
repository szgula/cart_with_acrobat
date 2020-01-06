from DQN.dqn_agent import Agent
import gym
import torch

env = gym.make('CartAcrobatDiscrete-v0')  # 'LunarLander-v2')
agent = Agent(state_size=6, action_size=3, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load(r'/Users/szymongula/Documents/Projects/ML/RL/FinalThesis/cart_with_acrobat/Version_2/DDPG/DQN/results/mean_checkpoint539.84.pth'))

for i in range(10):
    state = env.reset()
    sum_reward = 0
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        sum_reward += reward
        if done:
            break
    print(sum_reward)

env.close()