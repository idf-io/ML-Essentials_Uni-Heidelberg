import os
import pickle
import numpy as np

from settings import BOMB_POWER, COLS

# List of available actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

def setup(self):
    # Exploration parameters
    self.epsilon = 1.0  # Initial exploration rate
    self.learning_rate = 0.1  # Learning rate for Q-learning
    self.discount_factor = 0.9  # Discount factor for future rewards

    # Check if Q-table file exists, and load it if available
    if self.train or not os.path.isfile("my-saved-qtable-2.pkl"):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {}  # Initialize Q-table
    else:
        self.logger.info("Loading Q-table from saved state.")
        with open("my-saved-qtable-2.pkl", "rb") as file:
            self.q_table = pickle.load(file)

# Softmax function to convert Q-values into probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Select an action using epsilon-greedy or softmax exploration
def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS)

    state = state_to_features(game_state)

    if self.train:
        q_values = self.q_table.get(tuple(state), {a: 0.0 for a in ACTIONS})

        # Make sure all actions have Q-values
        for action in ACTIONS:
            if action not in q_values:
                q_values[action] = 0.0

        probabilities = softmax(list(q_values.values()))
        chosen_action = np.random.choice(ACTIONS, p=probabilities)
        return chosen_action
    else:
        # In testing mode, choose the action with the highest Q-value (exploitation)
        q_values = self.q_table.get(tuple(state), {a: 0.0 for a in ACTIONS})
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        # If there are multiple actions with the same max Q-value, select one randomly
        chosen_action = np.random.choice(best_actions)
        return chosen_action

# Convert game state to feature vector
def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    self_position = game_state["self"][3]
    new_field = field2bomb(game_state)
    new_field = field2coin(game_state, new_field)
    layers = 1

    features = []
    for x in range(self_position[0]-layers,self_position[0]+layers+1):
        for y in range(self_position[1]-layers,self_position[1]+layers+1):
            cell = new_field[x, y]
            if cell == 2:
                features.extend([1, 0, 0])
            elif cell == 1:
                features.extend([0, 1, 0])
            elif cell == 0:
                features.extend([0, 1, 1])
            elif cell == -1:
                features.extend([1, 1, 0])
            elif cell == 3:
                features.extend([0, 0, 1])
            else:
                features.extend([1, 1, 1])

    features.append(self_position[0])
    features.append(self_position[1])
    return np.array(features)

# Update the game field to include bomb explosion tiles
def field2bomb(game_state: dict, power=BOMB_POWER, board_size=COLS):

    """
    Convert the field array to include tiles where the explosion takes place next move.
    """

    bomb_mask = np.zeros([board_size, board_size])
    #0 dimension indicates the position of bomb,1 dimension indicates the time before explosion (0 right before explosion)
    for bomb in game_state["bombs"]:
        if bomb[1] == 0:
            bomb_mask[max(bomb[0][0]-power,0):min(bomb[0][0]+power+1,board_size),bomb[0][1]] = 1
            bomb_mask[bomb[0][0]][max(bomb[0][1]-power,0):min(bomb[0][1]+power+1,board_size)] = 1

    new_field = (game_state["field"] == 0) & np.array(bomb_mask, dtype=bool)
    new_field = np.where(new_field, 2, game_state["field"])

    return new_field

# Update the game field to include coin locations
def field2coin(game_state: dict, new_field):
    """
    :param game_state: current state of the game
    :param new_field: field with walls, crates and bombs
    :return: new_field with coin locations
    """
    coins = game_state["coins"]

    for coin in coins:
        if new_field[coin[0]][coin[1]] == 0 :
            new_field[coin[0]][coin[1]] = 3

    return new_field
