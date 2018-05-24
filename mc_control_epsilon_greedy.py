import matplotlib
import numpy as np
import sys
import time

from collections import defaultdict

from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """

    def policy_fn(observation):
        # Implement this!
        probs = np.zeros(nA)
        explore_prob = epsilon / nA
        greedy_prob = 1.0 - epsilon + explore_prob
        
        best_action = np.argmax(Q[observation])
        for i in range(nA):
            if i == best_action:
                probs[i] = greedy_prob
            else:
                probs[i] = explore_prob
        return probs
        
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    # Implement this!
    for i in range(num_episodes):
        if i % 1000 == 0:
            print("\rEpisode {}/{}.".format(i, num_episodes), end="")
            sys.stdout.flush()
            
        episode = []
        state = env.reset()
        while True:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action))
            state = next_state
            if done:
                for state_action in episode:
                    returns_sum[state_action] += reward
                    returns_count[state_action] += 1.0
                break

        for state_action, value in returns_sum.items():
            state = state_action[0]
            action = state_action[1]
            Q[state][action] = value / returns_count[state_action]
        
    return Q, policy


start_time = time.time()    
Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
print("--- %s seconds ---" % (time.time() - start_time))

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
