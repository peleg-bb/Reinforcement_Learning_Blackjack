import numpy as np
import gymnasium as gym
from tqdm import tqdm

NUM_SAMPLES = 100000


# import gym # env of adi
# from gym.envs.toy_text.blackjack import BlackjackEnv # env of adi

def transition_matrix(env, num_samples=100000):
    # Create dictionaries for counting transitions
    dictionary_count_all = {}
    dictionary_state_action_state = {}

    for _ in tqdm(range(num_samples)):
        # Initialize the game
        state = env.reset()
        state = state[0]
        done = False

        while not done:
            # Choose a random action
            action = env.action_space.sample()

            # Perform the action
            next_state, _, done, _, _ = env.step(action)

            state_action_next_state = (state, action, next_state)
            state_action = (state, action)

            # Update transition counts
            dictionary_state_action_state[state_action_next_state] = dictionary_state_action_state.get(state_action_next_state, 0) + 1
            dictionary_count_all[state_action] = dictionary_count_all.get(state_action, 0) + 1

            # Update the state
            state = next_state

    # Normalize transition counts
    for key, value in dictionary_state_action_state.items():
        state, action, next_state = key
        new_key = (state, action)
        dictionary_state_action_state[key] = value / dictionary_count_all.get(new_key, 1)

    print("Transition Matrix:")

    for key, value in dictionary_state_action_state.items():
        print("key: ", key, " value: ", value)

    print(len(dictionary_state_action_state))

    return dictionary_state_action_state


# Create Blackjack environment
# env = BlackjackEnv() # env of adi
env = gym.make("Blackjack-v1", sab=True) # env of Peleg
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=NUM_SAMPLES)

transition_matrix = transition_matrix(env, NUM_SAMPLES)

#policy = dict of state, (p to hit)


def initiate_policy():
    policy = {}
    already_used_state = {}
    for key, value in transition_matrix.items():
        state, action, next_state = key
        if not already_used_state.get(state, False):
            already_used_state[state] = True
            policy[state] = 0


def help_me_sum(values_of_policy, curr_state, action_to_do):
    sum_of_states = 0
    for key, value in transition_matrix.items():
        state, action, next_state = key
        if state == curr_state and action == action_to_do: # need to optimize this and iterate over the one that needed only
            sum_of_states += value*values_of_policy.get(next_state, 0)
    return sum_of_states


def approximate_policy_evaluation(policy, num_of_iterarion = 5):
    values_of_policy = {}
    for i in range(num_of_iterarion):
        already_used_state = {}
        for key, value in policy.items():
            action_to_do = np.random.choice([0, 1], p=[1 - value, value])
            _, reward, _, _, _ = env.step(action_to_do)
            values_of_policy[key] = reward + 1*help_me_sum(values_of_policy, key, action_to_do)


def apply_policy(policy):
    state = env.reset()
    done = False
    while not done: #check if done or terminated
        action_to_do = np.random.choice([0, 1], p=[1 - policy[state], policy[state]])
        state, reward, terminated, done, info = env.step(action_to_do)

# def value_function_q3(state, value):
#     if()



