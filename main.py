from __future__ import annotations
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

from BlackjackAgent import BlackjackAgent

LOSING_SCORE = 22

MAX_DEALER_SCORE = 11
MIN_DEALER_SCORE = 1
MAX_PLAYER_SCORE = 22
MIN_PLAYER_SCORE = 4

env = gym.make("Blackjack-v1", sab=True)
done = False
observation, info = env.reset()

# sample a random action from all valid actions
action = env.action_space.sample()
# action=1


# define a possible state to be a tuple of (player count, dealer count, usable ace)
possible_state = []
for i in range(MIN_PLAYER_SCORE, MAX_PLAYER_SCORE):
    for j in range(MIN_DEALER_SCORE, MAX_DEALER_SCORE):
        for k in [True, False]:
            possible_state.append((i, j, k))


def encode_state(state):
    """Encodes a state (player count, dealer count, usable ace) to a unique integer."""
    if state[0] > LOSING_SCORE:
        return (LOSING_SCORE + state[1] + (MAX_PLAYER_SCORE - MIN_PLAYER_SCORE) +
                state[2] + (MAX_PLAYER_SCORE-MIN_PLAYER_SCORE+MAX_DEALER_SCORE-MIN_DEALER_SCORE))
    else:
        return (state[0] + state[1] + (MAX_PLAYER_SCORE - MIN_PLAYER_SCORE) +
                state[2] + (MAX_PLAYER_SCORE - MIN_PLAYER_SCORE + MAX_DEALER_SCORE - MIN_DEALER_SCORE))


def add_transition(state, action, next_state):
    """Adds a transition from state to next_state with action to the transition matrix."""
    state, next_state = encode_state(state), encode_state(next_state)
    if action == 0:
        transition_matrix_action_0[state, next_state] += 1
    else:
        transition_matrix_action_1[state, next_state] += 1


state_to_int = {state: encode_state(state) for state in possible_state}

transition_matrix_action_0, transition_matrix_action_1 = np.zeros((len(possible_state), len(possible_state))), np.zeros(
    (len(possible_state), len(possible_state)))
print(transition_matrix_action_0.shape)

# for _ in range(3):
# observation, reward, terminated, truncated, info = env.step(action)
# print(observation, reward, terminated, truncated, info)
#

# observation=(24, 10, False)
# reward=-1.0
# terminated=True
# truncated=False
# info={}
# hyperparameters
learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1


class BlackjackAgent:
    def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)


agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # update the transition matrix
        add_transition(obs, action, next_obs)
        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)


        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

for i in range(len(transition_matrix_action_0)):
    if np.sum(transition_matrix_action_0[i]) != 0:
        transition_matrix_action_0[i] /= np.sum(transition_matrix_action_0[i])
    if np.sum(transition_matrix_action_1[i]) != 0:
        transition_matrix_action_1[i] /= np.sum(transition_matrix_action_1[i])

    #
    # rolling_length = 500
    # fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    # axs[0].set_title("Episode rewards")
    # # compute and assign a rolling average of the data to provide a smoother graph
    # reward_moving_average = (
    #         np.convolve(
    #             np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    #         )
    #         / rolling_length
    # )
    # axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    # axs[1].set_title("Episode lengths")
    # length_moving_average = (
    #         np.convolve(
    #             np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    #         )
    #         / rolling_length
    # )
    # axs[1].plot(range(len(length_moving_average)), length_moving_average)
    # axs[2].set_title("Training Error")
    # training_error_moving_average = (
    #         np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    #         / rolling_length
    # )
    # axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    # plt.tight_layout()
    # plt.show()




def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

#
# def create_plots(value_grid, policy_grid, title: str):
#     """Creates a plot using a value and policy grid."""
#     # create a new figure with 2 subplots (left: state values, right: policy)
#     player_count, dealer_count, value = value_grid
#     fig = plt.figure(figsize=plt.figaspect(0.4))
#     fig.suptitle(title, fontsize=16)
#
#     # plot the state values
#     ax1 = fig.add_subplot(1, 2, 1, projection="3d")
#     ax1.plot_surface(
#         player_count,
#         dealer_count,
#         value,
#         rstride=1,
#         cstride=1,
#         cmap="viridis",
#         edgecolor="none",
#     )
#     plt.xticks(range(12, 22), range(12, 22))
#     plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
#     ax1.set_title(f"State values: {title}")
#     ax1.set_xlabel("Player sum")
#     ax1.set_ylabel("Dealer showing")
#     ax1.zaxis.set_rotate_label(False)
#     ax1.set_zlabel("Value", fontsize=14, rotation=90)
#     ax1.view_init(20, 220)
#
#     # plot the policy
#     fig.add_subplot(1, 2, 2)
#     ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
#     ax2.set_title(f"Policy: {title}")
#     ax2.set_xlabel("Player sum")
#     ax2.set_ylabel("Dealer showing")
#     ax2.set_xticklabels(range(12, 22))
#     ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)
#
#     # add a legend
#     legend_elements = [
#         Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
#         Patch(facecolor="grey", edgecolor="black", label="Stick"),
#     ]
#     ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
#     return fig
#
#
# state values & policy with usable ace (ace counts as 11)
# value_grid, policy_grid = create_grids(agent, usable_ace=True)
# fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
# plt.show()
#
#



def transition_matrix(env, num_samples=100000000):
    # Create dictionaries for counting transitions
    dictionary_count_all = {}
    dictionary_state_action_state = {}

    for _ in range(num_samples):
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

   # print(dictionary_state_action_state)

# Create Blackjack environment


transition_matrix(env, num_samples=10000000)

transition_matrix(0)