import gymnasium as gym
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

num_samples = 100000

win_state = "win_state"
lose_state = "lose_state"
draw_state = "draw_state"


def transition_matrix(env, num_of_samples=num_samples):
    """
    This function creates a transition matrix for the given environment
    :param env: a gym environment for blackjack
    :param num_of_samples: number of samples to use for creating the transition matrix
    :return:
    """
    dictionary_count_all = {}
    dictionary_state_action_state = {}

    for _ in tqdm(range(num_of_samples)):
        # Initialize the game
        state = env.reset()
        state = state[0]
        done = False
        terminated = False

        while not done and not terminated:
            action = env.action_space.sample()
            next_state, reward, done, terminated, _ = env.step(action)
            if done or terminated:
                if reward == 0:
                    next_state = draw_state
                if reward == 1:
                    next_state = win_state
                if reward == -1:
                    next_state = lose_state

            state_action_next_state = (state, action, next_state)
            state_action = (state, action)
            dictionary_state_action_state[state_action_next_state] = dictionary_state_action_state.get(
                state_action_next_state, 0) + 1
            dictionary_count_all[state_action] = dictionary_count_all.get(state_action, 0) + 1
            state = next_state

    for key, value in dictionary_state_action_state.items():
        state, action, next_state = key
        new_key = (state, action)
        dictionary_state_action_state[key] = round(value / dictionary_count_all.get(new_key, 1), 3)

    return dictionary_state_action_state


env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_samples)

transition_matrix = transition_matrix(env, num_samples)

default_policy = {}
already_used_states = {}


def calculate_state_value(values_of_policy, curr_state, action_to_do):
    """
    This function calculates the value of a state according to a given policy
    :param values_of_policy: a dictionary of the value of each state
    :param curr_state: the current state for which we want to calculate the value
    :param action_to_do: the action for which we want to calculate
    :return: the value of the state
    """
    sum_of_states = 0
    for key, transition_probability in transition_matrix.items():
        state, action, next_state = key
        if state == curr_state and action == action_to_do:
            sum_of_states = sum_of_states + (transition_probability * values_of_policy.get(next_state, 0))
    return sum_of_states


def approximate_policy_evaluation(policy, num_of_iterations=8):
    """
    This function approximates the value of a given policy
    :param policy: a dictionary representing the policy action for each state
    :param num_of_iterations: max number of iterations to perform
    :return: values_of_policy: a dictionary of the value of each state
    """
    values_of_policy = {win_state: 1, lose_state: -1, draw_state: 0}

    for i in range(num_of_iterations):
        for state, action in policy.items():  # for all the states, including the 3 added
            state_value = calculate_state_value(values_of_policy, state, action)
            values_of_policy[state] = state_value
    return values_of_policy


def find_best_action(state, values_of_policy):
    """
    This function finds the best action for a given state according to the value of the states
    :return: stick or hit action
    """
    stick_value = calculate_state_value(values_of_policy, state, 0)
    hit_value = calculate_state_value(values_of_policy, state, 1)
    return 0 if stick_value > hit_value else 1


def greedy_policy_improvement(policy_value, states):
    """
    This function improves the policy according to the value of the states greedily
    :return: an improved policy
    """
    improved_policy = {}
    for state in states:
        improved_policy[state] = find_best_action(state, policy_value)
    return improved_policy


def policy_iteration():
    """
    This function performs the policy iteration algorithm
    :return: the optimal policy
    """
    _ = env.reset()
    policy = dict.fromkeys([state for state, action, next_state in transition_matrix.keys()], 0)
    improved_policy = dict.fromkeys([state for state, action, next_state in transition_matrix.keys()], 1)
    policy_values = []
    iterations = []
    num_of_iteration = 0
    while policy != improved_policy and num_of_iteration < num_samples:
        policy = improved_policy
        policy_value = approximate_policy_evaluation(policy)  # policy evaluation
        improved_policy = greedy_policy_improvement(policy_value, policy.keys())  # greedy policy improvement

        policy_values.append(sum(policy_value.values()) / len(policy_value.values()))  # adding to a list for the graph
        iterations.append(num_of_iteration)  # adding to a list for the graph
        num_of_iteration += 1

    plot_policy_graph(iterations, policy_values)  # plot the graph
    return improved_policy


def plot_policy_graph(iterations, policy_values):
    """
    This function plots the policy values over the iterations
    :param iterations: a list of the numbered iterations
    :param policy_values: a list of the policy values
    """
    plt.plot(iterations, policy_values)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Average Policy Value')
    plt.title('Policy Values over Iterations')
    plt.show()


def apply_policy(policy):
    """
    This function applies a given policy to an episode (or game) in the environment
    :param policy:
    :return:
    """
    state = env.reset()
    state = state[0]
    done = False
    terminated = False
    iteration_counter = 0
    reward_sum = 0
    while not done and not terminated:  # check if done or terminated
        action = policy.get(state, 0)  # policy[state]
        state, reward, terminated, done, info = env.step(action)  # apply the action
        reward_sum += reward
        iteration_counter += 1
    return reward_sum


def value_function_q3(state):
    """
    This function calculates the value of a state according to a given policy
    """
    return approximate_policy_evaluation(best_policy).get(state, 0)


best_policy = policy_iteration()
example_state = (13, 2, 0)
# print(f"The expected value of the state {example_state} is {value_function_q3(example_state)}")
num_of_success = 0
for episode in tqdm(range(num_samples)):
    episode_result = apply_policy(best_policy)  # apply the best policy to the environment, and get the result
    if episode_result > 0:
        num_of_success = num_of_success + 1

print("Game winning percentage of the optimal policy is %", 100 * num_of_success / num_samples)

# Convert the best_policy dictionary into a DataFrame
best_policy_df = (pd.DataFrame([(key[0], key[1], value) for (key, value) in best_policy.items() if (key[2] != 1)],
                               columns=['Player Score', 'Dealer Score', 'Action'])
                  .sort_values(by=['Player Score', 'Dealer Score']))
print("The optimal policy is: ")
print(best_policy_df)
