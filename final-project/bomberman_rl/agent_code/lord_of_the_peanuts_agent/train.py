from collections import namedtuple, deque
import pickle
import numpy as np
import events as e
from .callbacks import state_to_features, ACTIONS
from typing import List


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 10000
RECORD_ENEMY_TRANSITIONS = 1.0
PLACEHOLDER_EVENT = "PLACEHOLDER"
# Add custom events
MOVE_CLOSER_TO_COIN = "MOVE_CLOSER_TO_COIN"
MOVE_AWAY_FROM_COIN = "MOVE_AWAY_FROM_COIN"


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) #maintain a replay buffer (a deque of transitions) to store experiences.
    self.epsilon = 1.0
    self.epsilon_decay = 0.995
    self.min_epsilon = 0.4
    self.gamma = 0.9
    self.alpha = 0.1

def update_q_values(self, gamma):
    #sample batches from the replay buffer and update Q-values based on the Bellman equation.
    while self.transitions:
        transition = self.transitions.popleft()
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state

        if is_action_invalid(state, action):
            updated_q_value = -100.0
        else:
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

    with open("my-saved-qtable-14.pkl", "wb") as file:
        pickle.dump(self.q_table, file)

def is_action_invalid(state, action):
    if action == 'RIGHT' and all(state[21:24] == [1, 1, 0]):
        return True
    elif action == 'LEFT' and all(state[3:6] == [1, 1, 0]):
        return True
    elif action == 'UP' and all(state[9:12] == [1, 1, 0]):
        return True
    elif action == 'DOWN' and all(state[15:18] == [1, 1, 0]):
        return True
    else:
        return False
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    # Append custom events for moving towards/away from coin
    if e.COIN_COLLECTED not in events:
        if new_state[len(new_state)-1] < old_state[len(old_state)-1]:
            events.append(MOVE_CLOSER_TO_COIN)
        elif new_state[len(new_state)-1] > old_state[len(old_state)-1]:
            events.append(MOVE_AWAY_FROM_COIN)
    # Calculate reward based on events
    reward = reward_from_events(self, events)
    # add transitions to the replay buffer (store the state, action, next state, and reward)
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))
    # Gradually decrease epsilon
    self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    update_q_values(self, self.gamma)
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward))

def reward_from_events(self, events: List[str]) -> float:
    game_rewards = {
        e.COIN_COLLECTED: 200.0,
        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: 0,
        e.SURVIVED_ROUND: 0,
        e.COIN_FOUND: 0,
        e.GOT_KILLED: 0,
        e.CRATE_DESTROYED: 0,
        PLACEHOLDER_EVENT: 0,
        e.INVALID_ACTION: -50.0,
        MOVE_CLOSER_TO_COIN: 100.0,
        MOVE_AWAY_FROM_COIN: -100.0,
        e.WAITED: 0
    }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

