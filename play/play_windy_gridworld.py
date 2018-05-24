import sys

if "../" not in sys.path:
  sys.path.append("../") 

from lib.envs.windy_gridworld import WindyGridworldEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

env = WindyGridworldEnv()

print(env.reset())
print(env.nS)
print(env.nA)

for i in range(env.nS):
    print("state" + str(i) + ":")
    print(env.P[i])
env.render()

print(env.step(RIGHT))
env.render()

print(env.step(RIGHT))
env.render()

print(env.step(UP))
env.render()

#print(env.step(2))
#env.render()
#
#print(env.step(1))
#env.render()
#
#print(env.step(1))
#env.render()