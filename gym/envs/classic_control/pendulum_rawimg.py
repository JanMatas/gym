'''
This environment gives the raw image
as output instead of the low level state
values. The implementation is based on
PendulumEnv.
'''

from gym.envs.classic_control import pendulum
import numpy as np
from gym import spaces
import cv2


class PendulumRawImgEnv(pendulum.PendulumEnv):

    def __init__(self):
        super().__init__()
        self.drawer = DrawImage()
        self.raw_img = None
        self.obs = None
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(64,64,3), dtype='float32')


    def step(self, action):
        obs, rw, done, inf = super().step(action)
        cos_theta = obs[0]
        sin_theta = obs[1]
        self.obs = obs
        theta, thetadot = self.state
        self.raw_img = self.drawer.draw(cos_theta, sin_theta, thetadot)
        return self.raw_img, rw, done, inf

    def render(self, mode='human'):
        cos_theta = self.obs[0]
        sin_theta = self.obs[1]
        # self.raw_img = self.drawer.draw(cos_theta, sin_theta)
        if mode == 'rgb_array':
            return self.raw_img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.raw_img)
            return self.viewer.isopen

    def reset(self):
        obs = super().reset()
        cos_theta = obs[0]
        sin_theta = obs[1]
        # self.obs = obs
        theta, thetadot = self.state
        self.raw_img = self.drawer.draw(cos_theta, sin_theta, thetadot)
        return self.raw_img

    def get_state(self):
        return self._get_obs()

    def goalstate(self):
        return np.array([1, 0, 0])

    def goalobs(self):
        img = self.drawer.draw(1, 0, 0)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def __del__(self):
        self.close()

colors = {
        'black': np.array([0, 0, 0]),
        'red': np.array([214, 25, 28])
    }


class DrawImage:

    def __init__(self):
        self.height = 64  # Atari-like image size
        self.width = 64
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

    def draw(self, cos_theta, sin_theta, thetadot):
        self.__clear()
        self.__draw_rod(cos_theta, sin_theta, thetadot)
        return self.canvas

    # Draw the elements of the image

    def __clear(self):

        self.canvas[:, :, :] = 255

    def __draw_rod(self, cos_theta, sin_theta, thetadot):
        x0 = self.width // 2
        y0 = self.height // 2

        half_width = 3
        length_f = 30
        length_b = 3
        points = np.array([[-length_b, -half_width, 1 ],
                  [-length_b, half_width, 1],
                  [length_f, half_width, 1],
                  [length_f, -half_width, 1]]).T

        rot_points = np.matmul(np.array([[cos_theta, -sin_theta, x0],[sin_theta, cos_theta, y0]]), points).T

        pts = np.array([rot_points], np.int32)
        color = np.array((0 if thetadot < 0 else min(thetadot * 32, 255), 25, 0 if thetadot > 0 else min(-thetadot * 32, 255)))
        cv2.fillPoly(self.canvas, pts, color)

