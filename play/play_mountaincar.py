import sys
import numpy as np
from matplotlib import pyplot as plt

if "../" not in sys.path:
  sys.path.append("../") 
  
from lib.envs.mountain_car import MountainCarEnv

env = MountainCarEnv()

env.reset()

observation_examples = np.array([env.observation_space.sample() for x in range(10)])
print(observation_examples)
print(env.observation_space.high)
print(env.observation_space.low)
#
print(env.action_space)
action_examples = np.array([env.action_space.sample() for x in range(10)])
print(action_examples)

env.reset()
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

for i in range(10):
    next_state, reward, done, _ = env.step(0)
    print("next_state={} reward={} done={}".format(next_state, reward, done))

plt.figure()
plt.imshow(env.render(mode='rgb_array'))
env.render(close=True)