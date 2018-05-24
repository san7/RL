import matplotlib
import numpy as np

from collections import defaultdict

from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    
    # Implement this!
    for i in range(num_episodes):
        episode = []
        state = env.reset()
        while True:
            episode.append(state)
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                for state in episode:
                    returns_sum[state] += reward
                    returns_count[state] += 1.0
                break
    
    for k in returns_sum.keys():
        V[k] = returns_sum[k] / returns_count[k]
    
    return V  

def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return np.array([1.0, 0.0]) if score >= 16 else np.array([0.0, 1.0])


V_SMALL = mc_prediction(sample_policy, env, num_episodes=50000)
print(V_SMALL)
plotting.plot_value_function(V_SMALL, title="10,000 Steps")

#V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
#plotting.plot_value_function(V_500k, title="500,000 Steps")
