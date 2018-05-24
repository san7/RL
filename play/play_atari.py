import gym
import numpy as np

env = gym.envs.make("Breakout-v0")

env.reset()

observation_examples = np.array([env.observation_space.sample() for x in range(10)])
print(observation_examples)