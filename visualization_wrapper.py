import torch
import numpy as np
import matplotlib.pyplot as plt
from env_model import CartAcrobat


from ddpg_agent import Agent

#import scipy.integrate as integrate
import matplotlib.animation as animation


np.set_printoptions(precision=4)


agent = Agent(state_size=6, action_size=1, random_seed=2, num_agents=1)
#agent.actor_local.load_state_dict(torch.load('colab_results/checkpoint_actor_force50_191.85710679936776_per_250.pth', map_location=lambda storage, loc: storage))
#agent.critic_local.load_state_dict(torch.load('colab_results/checkpoint_critic_force50_191.85710679936776_per_250.pth', map_location=lambda storage, loc: storage))
#agent.actor_target.load_state_dict(torch.load('colab_results/checkpoint_actor_force50_191.85710679936776_per_250.pth', map_location=lambda storage, loc: storage))
#agent.critic_target.load_state_dict(torch.load('colab_results/checkpoint_critic_force50_191.85710679936776_per_250.pth', map_location=lambda storage, loc: storage))
agent.actor_local.load_state_dict(torch.load('checkpoint_actor_force50_energy_.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic_force50_energy_.pth'))
agent.actor_target.load_state_dict(torch.load('checkpoint_actor_force50_energy_.pth'))
agent.critic_target.load_state_dict(torch.load('checkpoint_critic_force50_energy_.pth'))


pendulum = CartAcrobat(m=(0.2, 0.2, 0.2), b=(0.2, 0.1, 0.1), rail_lims=(-200, 200), num_instances=1)

reward_sum = 0
counter = 0

def visualize():
    # ------------------------------------------------------------
    # set up initial state and global variables

    Q = np.random.sample((1, pendulum.n_q))
    Q[0][0] = 0
    Q[0][1] = np.deg2rad(180)
    Q[0][2] = np.deg2rad(180)
    Q[0][3] = 0
    Q[0][4] = 0
    Q[0][5] = 0

    U = agent.act(Q, add_noise=False)
    pendulum.update_state(Q[0])

    dt = 0.01

    # ------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-5, 5), ylim=(-2.5, 2.5))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    base, = ax.plot([], [], '-', lw=2)
    wheel, = ax.plot([], [], marker='o', ms=20)
    time_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
    energy_text = ax.text(0.02, 0.55, '', transform=ax.transAxes)
    reward_text = ax.text(0.4, 0.80, '', transform=ax.transAxes)
    counter_text = ax.text(0.8, 0.80, '', transform=ax.transAxes)


    def init():
        """initialize animation"""
        line.set_data([], [])
        base.set_data([-2, 2], [-0.17, -0.17])
        wheel.set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        reward_text.set_text('')
        counter_text.set_text('')
        return line, time_text, energy_text, reward_text


    def animate(i):
        """perform animation step"""
        global pendulum, dt, agent, reward_sum, counter
        state = np.expand_dims(pendulum.state, axis=0)
        U = agent.act(state, add_noise=False)
        _, reward, _, _ = pendulum.step(U, 0.01)
        pp = pendulum.get_position_to_plot()
        line.set_data(*pp)
        wheel.set_data(pp[0][0],pp[1][0])
        time_text.set_text('time = %.1f' % pendulum.time_elapsed)
        ene = pendulum.energy()
        energy_text.set_text(f'energy: \n {ene[0]} \n {ene[1]}')
        reward_sum += reward
        reward_text.set_text(f'reward: \n {reward_sum}')
        counter += 1
        counter_text.set_text(f'counter: \n {counter}')
        return line, wheel, time_text, energy_text, reward_text, counter_text


    # choose the interval based on dt and the time to animate one step
    from time import time

    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, frames=300,
                                  interval=interval, blit=True, init_func=init)


    plt.show()


if __name__ == "__main__":
    visualize()