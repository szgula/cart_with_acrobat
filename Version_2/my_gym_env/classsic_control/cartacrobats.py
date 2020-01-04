import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartAcrobatsEnv(gym.Env):   #CartDoublePoleEnv
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, num_env=2):
        self.num_env = num_env
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole_1 = 0.1
        self.masspole_2 = 0.1
        self.total_mass = (self.masspole_1 + self.masscart + self.masspole_2)
        self.length_1 = 0.1 # actually half the pole's length
        self.length_2 = 0.1  # actually half the pole's length
        self.polemass_length_1 = (self.masspole_1 * self.length_1)
        self.polemass_length_2 = (self.masspole_2 * self.length_2)
        self.force_mag = 30.0
        self.tau = 0.01  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0
        self.friction = np.array([[0.2], [0.01], [0.01]])

        # Angle at which to fail the episode
        self.theta_1_threshold_radians = math.pi
        self.theta_2_threshold_radians = math.pi
        self.x_threshold = 4.8

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_1_threshold_radians * 2,
            np.finfo(np.float32).max,
            self.theta_2_threshold_radians * 2,
            np.finfo(np.float32).max
        ])

        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )

        self.seed()
        self.viewer = None
        self.state = None
        self.done = None

        self.steps_beyond_done = None
        #self.spec.env.spec.max_episode_steps = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dstate_dt(self, force, dt=0.05):
        q = self.state.T
        theta1 = q[2]
        theta2 = q[4]
        dp = q[1]
        dtheta1 = q[3]
        dtheta2 = q[5]

        # TODO: check if this is needed
        #b = [min(my_iter, 1 / dt - 1e-10) for my_iter in b]

        M = np.zeros((self.num_env, 3, 3))
        M[:, 0, 0] = np.sum(self.masscart)
        M[:, 0, 1] = self.length_1 * (self.masspole_1 + self.masspole_2) * np.cos(theta1)
        M[:, 0, 2] = self.masspole_2 * self.length_2 * np.cos(theta2)

        M[:, 1, 0] = M[:, 0, 1]
        M[:, 1, 1] = (self.length_1 ** 2) * (self.masspole_1 + self.masspole_2)
        M[:, 1, 2] = self.length_1 * self.length_2 * self.masspole_2 * np.cos(theta1 - theta2)

        M[:, 2, 0] = M[:, 0, 2]
        M[:, 2, 1] = M[:, 1, 2]
        M[:, 2, 2] = (self.length_2 ** 2) * self.masspole_2

        P1 = np.zeros((3, self.num_env))
        P1[0, :] = ((self.length_1 * (self.masspole_1 + self.masspole_2) * (dtheta1 ** 2) * np.sin(theta1))
                       + (self.masspole_2 * self.length_2 * (dtheta2 ** 2) * np.sin(theta2)))
        P1[1, :] = ((-1) * self.length_1 * self.length_2 * self.masspole_2 * (dtheta2 ** 2) * np.sin(theta1 - theta2)
                       + self.gravity * (self.masspole_1 + self.masspole_2) * self.length_1 * np.sin(theta1))
        P1[2, :] = (self.length_1 * self.length_2 * self.masspole_2 * (dtheta1 ** 2) * np.sin(theta1 - theta2)
                       + self.gravity * self.length_2 * self.masspole_2 * np.sin(theta2))

        P2 = self.friction * np.array([dp, dtheta1, dtheta2])

        P3 = np.zeros((3, self.num_env))
        P3[0, :] = force

        P = P1 - P2 + P3

        acc = np.zeros((3, self.num_env))
        for i in range(M.shape[0]):
            acc[:, i] = np.matmul(np.linalg.inv(M[i]), P.T[i])
        return acc

    @staticmethod
    def norm_angle(angle):
        angle[angle > np.pi] -= 2*np.pi
        angle[angle < -np.pi] += 2 * np.pi
        return angle


    def step(self, action):
        if np.any(self.done):
            instances_done = np.where(self.done)[0]
            self.reset_single(instances_done)
        state = self.state.T
        x, x_dot, theta_1, theta_dot_1, theta_2, theta_dot_2 = state
        force = self.force_mag * action.clip(-1.0, 1.0)

        accel = self.dstate_dt(force, dt=self.tau)

        x_next = x + self.tau * x_dot + 0.5 * accel[0] * self.tau * self.tau
        theta_1_next = theta_1 + self.tau * theta_dot_1 + 0.5 * accel[1] * self.tau * self.tau
        theta_1_next = self.norm_angle(theta_1_next)
        theta_2_next = theta_2 + self.tau * theta_dot_2 + 0.5 * accel[2] * self.tau * self.tau
        theta_2_next = self.norm_angle(theta_2_next)

        x_dot_next = x_dot + self.tau * accel[0]
        theta_dot_1_next = theta_dot_1 + self.tau * accel[1]
        theta_dot_2_next = theta_dot_2 + self.tau * accel[2]

        self.state = np.array((x_next, x_dot_next, theta_1_next, theta_dot_1_next, theta_2_next, theta_dot_2_next)).T
        done = np.logical_or(x < -self.x_threshold, x > self.x_threshold)
        in_reward_range = np.logical_and(abs(theta_1_next) < np.deg2rad(15), abs(theta_2_next) < np.deg2rad(15))
        reward = np.ones((self.num_env,)) * ~done * in_reward_range
        self.done = done

        return self.state, reward, done, {}


    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.num_env, 6))
        self.state[:, 2] = self.np_random.uniform(low=-np.pi, high=np.pi, size=(self.num_env,))
        self.state[:, 4] = self.np_random.uniform(low=-np.pi, high=np.pi, size=(self.num_env,))
        self.steps_beyond_done = None
        self.done = np.array([False]*self.num_env)
        return np.array(self.state)

    def reset_single(self, instance):
        num_instances_to_reset = len(instance)
        self.state[instance] = self.np_random.uniform(low=-0.05, high=0.05, size=(num_instances_to_reset, 6))
        self.state[instance, 2] = self.np_random.uniform(low=-np.pi, high=np.pi, size=(num_instances_to_reset,))
        self.state[instance, 4] = self.np_random.uniform(low=-np.pi, high=np.pi, size=(num_instances_to_reset,))
        self.done[instance] = np.array([False]*num_instances_to_reset)

    def render(self, mode='human'):

        screen_width = 500
        screen_height = 300

        world_width = self.x_threshold*2
        scale = screen_width / world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0
        x = self.state

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l,r,t,b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            pole_2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole_2.set_color(.8, .6, .4)
            self.poletrans_2 = rendering.Transform(translation=(0, axleoffset))
            pole_2.add_attr(self.poletrans_2)

            self.viewer.add_geom(pole_2)

            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)

            self.axle_2 = rendering.make_circle(polewidth / 2)
            self.axle_2.add_attr(self.poletrans_2)
            #self.axle.add_attr(self.carttrans)
            self.axle_2.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle_2)

            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None
        cartx = x[0,0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[0, 2])

        self.poletrans_2.set_translation(cartx + np.sin(x[0, 2]) * polelen, carty + np.cos(x[0, 2]) * polelen)
        self.poletrans_2.set_rotation(-x[0, 4])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()