import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from chainerrl.wrappers.atari_wrappers import ScaledFloatFrame
from gym.wrappers import FrameStack, GrayScaleObservation, Monitor
from wrapper import *
import matplotlib.pyplot as plt
import numpy as np

class MyMarioEnv:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.env = GrayScaleObservation(self.env, True)
        self.env = Downsample(self.env, 2)
        self.env = ScaledFloatFrame(self.env)

        self.before_info = None

        self.skip_frame = 1

    def reset(self):
        state = self.env.reset()
        return state[np.newaxis, :, :]

    def step(self, action):
        for _ in range(self.skip_frame + 1):
            state, _reward, done, info = self.env.step(action)
            if done:
                break
        reward = 0
        if info["flag_get"]:
            reward += info["score"] / 100
            print("GOAL (reward) : ", reward)
        self.before_info = info
        return state[np.newaxis, :, :], reward, done, info

    def render(self):
        self.env.render()

    def render_state(self):
        plt.imshow(self.last_state, cmap = "gray")
        plt.show()


    
