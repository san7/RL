import sys

if "../" not in sys.path:
  sys.path.append("../") 

from lib.envs.cliff_walking import CliffWalkingEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

env = CliffWalkingEnv()

print(env.reset())
env.render()

print(env.step(UP))
env.render()

print(env.step(RIGHT))
env.render()

print(env.step(RIGHT))
env.render()

print(env.step(DOWN))
env.render()
