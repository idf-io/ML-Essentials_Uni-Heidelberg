import os
import pickle
import numpy as np

from settings import BOMB_POWER, COLS

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

def setup(self):
    self.epsilon = 1.0
    self.learning_rate = 0.1  # Learning rate for Q-learning
    self.discount_factor = 0.9  # Discount factor for future rewards
    if self.train or not os.path.isfile("my-saved-qtable.pkl"):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {} # Initialize Q-table
    else:
        self.logger.info("Loading Q-table from saved state.")
        with open("my-saved-qtable.pkl", "rb") as file:
            self.q_table = pickle.load(file)

def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS)

    state = state_to_features(game_state)
    if self.train and np.random.rand() < self.epsilon:
        return np.random.choice(ACTIONS)

    q_values = self.q_table.get(tuple(state), {a: 0.0 for a in ACTIONS})
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]

    return np.random.choice(best_actions)

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    self_position = game_state["self"][3]

    features = []
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            cell = field[x, y]
            if cell == -1:
                features.extend([1, 0, 0])
            elif cell == 1:
                features.extend([0, 1, 0])
            else:
                features.extend([0, 0, 1])

    features.append(self_position[0])
    features.append(self_position[1])

    return np.array(features)


def field2bomb(field, bombs, power=BOMB_POWER, board_size=COLS):

    """
    Convert the field array to include tiles where the explosion takes place next move.
    """

    bomb_mask = np.zeros([board_size, board_size])
    #0 dimension indicates the position of bomb,1 dimension indicates the time before explosion (0 right before explosion)
    for bomb in bombs:
        if bomb[1] == 0:
            bomb_mask[max(bomb[0][0]-power,0):min(bomb[0][0]+power+1,board_size),bomb[0][1]] = 1
            bomb_mask[bomb[0][0]][max(bomb[0][1]-power,0):min(bomb[0][1]+power+1,board_size)] = 1

    new_field = (field == 0) & np.array(bomb_mask, dtype=bool)
    new_field = np.where(new_field, 2, field)

    return new_field