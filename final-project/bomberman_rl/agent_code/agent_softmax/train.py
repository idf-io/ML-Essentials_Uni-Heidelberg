from collections import namedtuple, deque
import pickle
import numpy as np
import events as e
from .callbacks import state_to_features
from typing import List

# Define a named tuple to represent a transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 5
RECORD_ENEMY_TRANSITIONS = 1.0
PLACEHOLDER_EVENT = "PLACEHOLDER"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


def setup_training(self):
    # Initialize replay buffer to store experiences
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Q-learning hyperparameters
    self.learning_rate = 0.01
    self.discount_factor = 0.95
    self.epsilon = 1.0  # Exploration factor
    self.epsilon_decay = 0.9  # Decay rate for exploration
    self.min_epsilon = 0.1  # Minimum exploration factor
    self.gamma = 0.95  # Discount factor for rewards in the Bellman equation
    self.alpha = 0.1  # Learning rate for Q-value updates


def update_q_values(self, gamma):
    # Sample batches from the replay buffer and update Q-values based on the Bellman equation.
    while self.transitions:
        transition = self.transitions.popleft()
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state

        q_value = self.q_table.get(tuple(state), {}).get(action, 0.0)

        if next_state is None:
            target = reward
        else:
            next_q_values = self.q_table.get(tuple(next_state), {a: 0.0 for a in ACTIONS})
            max_next_q = max(next_q_values.values())
            target = reward + gamma * max_next_q

        # Update Q-value using the Q-learning update rule
        updated_q_value = q_value + self.alpha * (target - q_value)

        # Update Q-table
        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = {}
        self.q_table[tuple(state)][action] = updated_q_value

    # Save updated Q-table
    with open("my-saved-qtable-2.pkl", "wb") as file:
        pickle.dump(self.q_table, file)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Calculate reward based on events
    reward = reward_from_events(self, events)
    # Add transitions to the replay buffer (store the state, action, next state, and reward)
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))
    # Gradually decrease epsilon for exploration
    self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    update_q_values(self, self.gamma)
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward))


def reward_from_events(self, events: List[str]) -> float:
    # Define rewards for different events
    game_rewards = {
        e.COIN_COLLECTED: 5.0,
        e.KILLED_OPPONENT: 3.0,
        e.KILLED_SELF: -2.0,
        e.SURVIVED_ROUND: 1.0,
        e.COIN_FOUND: 1.0,
        e.GOT_KILLED: -2.0,
        e.CRATE_DESTROYED: 1.0,
        PLACEHOLDER_EVENT: -0.1,
        e.INVALID_ACTION: -0.1
    }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
