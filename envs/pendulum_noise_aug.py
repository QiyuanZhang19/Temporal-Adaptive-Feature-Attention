import numpy as np
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv


class PendulumAug(PendulumEnv):
    def __init__(self):
        super(PendulumAug, self).__init__()
        high = np.array([1., 1., self.max_speed, 10, 10], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32)

    def _get_obs(self):
        theta, thetadot = self.state
        noise1 = np.random.randn(1)[0] * 10
        noise2 = np.random.randn(1)[0] * 10
        return np.array([np.cos(theta), np.sin(theta), thetadot, noise1, noise2])













