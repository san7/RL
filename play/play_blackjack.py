import sys

if "../" not in sys.path:
  sys.path.append("../") 

from lib.envs.blackjack import BlackjackEnv

env = BlackjackEnv()

def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

def strategy(observation):
    score, dealer_score, usable_ace = observation
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise
    return 0 if score >= 18 else 1

for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        print_observation(observation)
        action = strategy(observation)
        print("Taking action: {}".format( ["Stick", "Hit"][action]))
        observation, reward, done, _ = env.step(action)
        print("reward={}".format(reward))
        if done:
            print_observation(observation)
            print("Game end. Reward: {}\n".format(float(reward)))
            break