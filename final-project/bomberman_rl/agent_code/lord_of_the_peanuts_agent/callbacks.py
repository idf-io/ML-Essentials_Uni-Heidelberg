import os
import pickle
import numpy as np
import logging
from settings import BOMB_POWER, COLS

# from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    # self.self_positions = deque(maxlen=30)
    if self.args.command_name == "play":
        qtable_load = self.args.qtable

    if self.train and not os.path.isfile(qtable_load):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {}  # Initialize Q-table
    else:
        self.logger.info("Loading Q-table from saved state.")
        with open(qtable_load, "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS)

    state = state_to_features(self, game_state)
    if self.train and np.random.rand() < self.epsilon:
        return np.random.choice(ACTIONS)

    q_values = self.q_table.get(tuple(state), {a: 0.0 for a in ACTIONS})
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]

    return np.random.choice(best_actions)


def state_to_features_ori(game_state: dict) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    self_position = game_state["self"][3]
    new_field = field2bomb(game_state)
    new_field = field2coin(game_state, new_field)
    layers = 1

    features = []
    for x in range(self_position[0] - layers, self_position[0] + layers + 1):
        for y in range(self_position[1] - layers, self_position[1] + layers + 1):
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


def state2position_features_rings(game_state, agent_position, layers: int = 1) -> list:
    """
    Convert the game_state to an array of features describing the agent's surroundings
    Input:
    - game_state: np.array
    - agent_position: list
    Output:
    - features: list (len=38)
    """

    features = []

    for x in range(agent_position[0] - layers, agent_position[0] + layers + 1):
        for y in range(agent_position[1] - layers, agent_position[1] + layers + 1):

            assert not (x < 0 or x >= game_state.shape[0] or y < 0 or y >= game_state.shape[1]), \
                (f"\n"
                 f"x>=0 and x<game_state.shape[0]:\n"
                 f"\tx = {x}\n"
                 f"\tgame_state.shape[0] = {game_state.shape[0]}\n"
                 f"y>=0 and y<game_state.shape[0]:\n"
                 f"\ty = {y}\n"
                 f"\tgame_state.shape[1] = {game_state.shape[1]}")

            cell = game_state[x, y]

            # Create feature matrix: Code surrounding tiles' states in binary

            if cell == 2:
                # Bomb
                features.extend([1, 0, 0])

            elif cell == 1:
                # Crate
                features.extend([0, 1, 0])

            elif cell == 0:
                # Empty tile
                features.extend([0, 1, 1])

            elif cell == -1:
                # Wall
                features.extend([1, 1, 0])

            elif cell == 3:
                # Coin
                features.extend([0, 0, 1])

            else:
                assert False, f"Cell value {cell} not expected nor covered in map above."

    # Remove centre cell
    centre_cell = int(len(features) / 2)
    del features[centre_cell - 1: centre_cell + 2]

    return features


def state_to_features(self, game_state: dict) -> np.array:
    if game_state is None:
        return None

    # reset the field to accelerate the algorithm
    field = game_state["field"]
    self_position = game_state["self"][3]
    new_field = field2bomb(game_state)
    new_field = field2coin(game_state, new_field)
    # self.self_positions.append(self_position)

    # !!think more about the features, such as vector to all coins

    # reduce the feature
    # layers=0 means that only consider the information of actual field

    layers = 1

    features = state2position_features_rings(game_state=new_field,
                                             agent_position=self_position,
                                             layers=layers)

    # Append features of nearest coin location (x, y) coord in binary form (respectively 15 possibilities so 2^4 -> 4 features for each coordinate.
    # E.g. max: 15 = b'1111'
    #      min:  0 = b'0000'
    # Mannhattan distance

    """    
    if check_if_agent_stuck(self.self_positions):
        features.extend([1])
    else:
        features.extend([0])
    """

    coins = np.array(game_state['coins'], dtype="int")
    if len(coins) > 0:
        position = np.array(self_position, dtype="int")
        nearest_coin = coins[np.argmin(np.sum(np.abs(coins - position), axis=1))]

        x = list("{0:04b}".format(nearest_coin[0]))
        x = [int(i) for i in x]
        y = list("{0:04b}".format(nearest_coin[1]))
        y = [int(i) for i in y]

        nearest_coin_bin = [*x, *y]
        features.extend(nearest_coin_bin)
        # Append agent's position in binary
        x = list("{0:04b}".format(self_position[0]))
        x = [int(i) for i in x]
        y = list("{0:04b}".format(self_position[1]))
        y = [int(i) for i in y]

        self_position_bin = [*x, *y]
        features.extend(self_position_bin)

        # Calculate the distance between nearest coin and agent
        distance = np.abs(nearest_coin[0] - self_position[0]) + np.abs(
            nearest_coin[1] - self_position[1])

        features.extend([distance])

    else:
        features.extend([0] * 17)

    return np.array(features)


def field2bomb(game_state: dict, power=BOMB_POWER, board_size=COLS):
    """
    Convert the field array to include tiles where the explosion takes place next move.
    """

    bomb_mask = np.zeros([board_size, board_size])
    # 0 dimension indicates the position of bomb,1 dimension indicates the time before explosion (0 right before explosion)
    for bomb in game_state["bombs"]:
        if bomb[1] == 0:
            bomb_mask[max(bomb[0][0] - power, 0):min(bomb[0][0] + power + 1, board_size), bomb[0][1]] = 1
            bomb_mask[bomb[0][0]][max(bomb[0][1] - power, 0):min(bomb[0][1] + power + 1, board_size)] = 1

    new_field = (game_state["field"] == 0) & np.array(bomb_mask, dtype=bool)
    new_field = np.where(new_field, 2, game_state["field"])

    return new_field


def field2coin(game_state: dict, new_field):
    """
    :param game_state: current state of the game
    :param new_field: field with walls, crates and bombs
    :return: new_field with coin locations
    """
    coins = game_state["coins"]

    for coin in coins:
        if new_field[coin[0]][coin[1]] == 0:
            new_field[coin[0]][coin[1]] = 3

    return new_field


"""
def check_if_agent_stuck(actions):
    if len(actions) == 30:
        unique_set = set(actions)
        unique_count = len(unique_set)

        if unique_count <= 2:
            return True
    else:
        return False
"""
