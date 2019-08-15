
import numpy as np
np.set_printoptions(precision=4)


class CartAcrobat:
    def __init__(self, l1=1, l2=1, m=(0.3, 0.2, 0.2), b=(0.2, 0.01, 0.01), g=9.81, rail_lims=(-3, 3),
                 force_lims=(-20, 20), num_instances=1):
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
        Ek = np.array([0.5 * self.m[0] * (self.state[:, 3] ** 2),
                       self.m[1] * ((self.state[:, 3] + self.l1 * self.state[:, 4] * np.cos(self.state[:, 1])) ** 2 +
                                    (0 - self.l1 * self.state[:, 4] * np.sin(self.state[:, 1])) ** 2),
                       self.m[2] * ((self.state[:, 3] + self.l1 * self.state[:, 4] * np.cos(
                           self.state[:, 1]) + self.l2 * self.state[:, 5] * np.cos(self.state[:, 2])) ** 2 +
                                    (0 - self.l1 * self.state[:, 4] * np.sin(self.state[:, 1]) - self.l2 * self.state[:,
                                                                                                           5] * np.sin(
                                        self.state[:, 2])) ** 2)]).transpose()
        Ep = np.array([np.zeros((self.num_of_instances,)),
                       self.m[1] * self.l1 * np.cos(self.state[:, 1]),
                       self.m[2] * (self.l1 * np.cos(self.state[:, 1]) + self.l2 * np.cos(
                           self.state[:, 2]))]).transpose()

        return Ek, Ep

    def get_max_pot_energy(self):
        return self.max_pot_1, self.max_pot_2

    def get_norm_hight(self):
        q = self.state
        h1 = np.cos(q[:, 1])
        h2 = (self.l1 * h1 + self.l2 * np.cos(q[:, 2])) / (self.l1 + self.l2)
        return h1, h2

    def dstate_dt(self, u, dt=0.05):
        q = self.state
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)
        theta1 = q[:, 1]
        theta2 = q[:, 2]
        dp = q[:, 3]
        dtheta1 = q[:, 4]
        dtheta2 = q[:, 5]

        b = [min(my_iter, 1 / dt - 1e-10) for my_iter in self.b]

        M = np.zeros((self.num_of_instances, 3, 3))
        M[:, 0, 0] = np.sum(self.m)
        M[:, 0, 1] = self.l1 * (self.m[1] + self.m[2]) * np.cos(theta1)
        M[:, 0, 2] = self.m[2] * self.l2 * np.cos(theta2)

        M[:, 1, 0] = M[:, 0, 1]
        M[:, 1, 1] = (self.l1 ** 2) * (self.m[1] + self.m[2])
        M[:, 1, 2] = self.l1 * self.l2 * self.m[2] * np.cos(theta1 - theta2)

        M[:, 2, 0] = M[:, 0, 2]
        M[:, 2, 1] = M[:, 1, 2]
        M[:, 2, 2] = (self.l2 ** 2) * self.m[2]

        P1 = np.zeros((self.num_of_instances, 3, 1))
        P1[:, 0, 0] = ((self.l1 * (self.m[1] + self.m[2]) * (dtheta1 ** 2) * np.sin(theta1))
                       + (self.m[2] * self.l2 * (dtheta2 ** 2) * np.sin(theta2)))
        P1[:, 1, 0] = ((-1) * self.l1 * self.l2 * self.m[2] * (dtheta2 ** 2) * np.sin(theta1 - theta2)
                       + self.g * (self.m[1] + self.m[2]) * self.l1 * np.sin(theta1))
        P1[:, 2, 0] = (self.l1 * self.l2 * self.m[2] * (dtheta1 ** 2) * np.sin(theta1 - theta2)
                       + self.g * self.l2 * self.m[2] * np.sin(theta2))

        P2 = b * q[:, self.n_d:]
        P2 = P2.reshape((self.num_of_instances, 3, 1))

        P3 = np.zeros((self.num_of_instances, 3, 1))
        P3[:, 0, 0] = u.reshape(self.num_of_instances)

        P = P1 - P2 + P3

        d_y = np.matmul(np.linalg.inv(M), P)
        return (d_y.reshape((self.num_of_instances, 3)))

    # @timeit
    def step(self, u, dt, disturb=0.0):
        done = False
        q = self.state
        q = np.array(q, dtype=np.float64)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)

        # Enforce input saturation and add disturbance
        u_act = np.clip(np.float64(u), self.force_lims[0], self.force_lims[1]) + disturb

        # Get numpy views of pose and twist, and compute accel
        pose = np.copy(q[:, :self.n_d])
        twist = np.copy(q[:, self.n_d:])
        accel = self.dstate_dt(u, dt=dt)

        # Partial-Verlet integrate state and enforce rail constraints
        pose_next = pose + dt * twist + 0.5 * (dt ** 2)
        done = np.logical_or(q[:, 0] < self.rail_lims[0], q[:, 0] > self.rail_lims[1])
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


