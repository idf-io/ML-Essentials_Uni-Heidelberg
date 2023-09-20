import os
import pickle
import numpy as np
import logging
from settings import BOMB_POWER, COLS
from datetime import datetime

from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    # self.self_positions = deque(maxlen=30)
    self.bombs_prev_step = []
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
    """
    # function to be continued: save bombs from previous round bc explosion lasts for an extra step
    self.bombs_prev_step = []
    for bomb in game_state['bombs']:
        if bomb[1] == 0:
            self.bombs_prev_step.append(bomb)
    self.bombs_prev_step = game_state['bombs']
    print(game_state['step'], self.bombs_prev_step)
    """
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


def state2position_features_cross(game_state, agent_position) -> list:
    """
    Convert the game_state to an array of features describing the agent's surroundings.
    Only take the tiles above, below, left and right from the agent.
    Input:
    - game_state: np.array
    - agent_position: list
    Output:
    - features: list (len=15=5*3)
    """

    features = []

    # Define relevant cells
    cells = (
        (agent_position[0] - 1, agent_position[1]),
        (agent_position[0] + 1, agent_position[1]),
        (agent_position[0], agent_position[1]),
        (agent_position[0], agent_position[1] - 1),
        (agent_position[0], agent_position[1] + 1)
    )

    for c in cells:

        assert not (c[0] < 0 or c[0] >= game_state.shape[0] or c[1] < 0 or c[1] >= game_state.shape[1]), \
            (f"\n"
             f"c[0]>=0 and c[0]<game_state.shape[0]:\n"
             f"\tc[0] = {c[0]}\n"
             f"\tgame_state.shape[0] = {game_state.shape[0]}\n"
             f"c[1]>=0 and c[1]<game_state.shape[0]:\n"
             f"\tc[1] = {c[1]}\n"
             f"\tgame_state.shape[1] = {game_state.shape[1]}")

        cell = game_state[c]

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

    return features


def state2position_features_rings(game_state, agent_position, layers: int = 1) -> list:
    """
    Convert the game_state to an array of features describing the agent's surroundings
    Input:
    - game_state: np.array
    - agent_position: list
    - layers: amount of tiles to consider in a radial distance from the agent
    Output:
    - features: list (len=24=3*8)
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
    # centre_cell = int(len(features) / 2)
    # del features[centre_cell - 1: centre_cell + 2]

    return features


# TO-DO:  make function for when we test hyperparameters
#     position = np.array(self_position, dtype="int")
#     nearest_coin = coins[np.argmin(np.sum(np.abs(coins - position), axis=1))]


def array2graph(array, nogo: list = [-1, 1]) -> dict:
    graph = {}

    for x in range(array.shape[0]):
        for y in range(array.shape[1]):

            if array[x, y] not in nogo:

                node = (x, y)
                edges = []

                cells = (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1)
                )

                for c in cells:

                    if c[0] < 0 or c[0] > array.shape[0]:
                        continue

                    if c[1] < 0 > array.shape[1]:
                        continue

                    if array[c[0], c[1]] in nogo:
                        continue

                    edges.append(c)

                graph[node] = tuple(edges)

    return graph


def breadth_first_search(graph: dict, start: tuple, end: tuple, nr_nodes: int):
    visited = deque(maxlen=nr_nodes)
    queue = deque(maxlen=nr_nodes)
    prev = {}

    visited.append(start)
    queue.append(start)

    while queue:

        node = queue.popleft()

        if node == end:
            return prev

        for edge in graph[node]:

            if edge not in visited:
                queue.append(edge)
                visited.append(edge)
                prev[edge] = node

    return {}


def reconstruct_path(start: tuple, end: tuple, prev: dict):
    shortest_path = []

    i = end

    while True:

        shortest_path.append(i)  # Add the current node to the path
        i = prev[i]  # Move to the predecessor (parent) node
        if i == start:
            shortest_path.append(i)
            break

    shortest_path.reverse()

    if shortest_path[0] == start:
        return tuple(shortest_path)

    return ()


def get_distance_and_move(start: tuple, end: tuple, graph: dict, nr_nodes: int):
    prev = breadth_first_search(graph, start, end, nr_nodes)
    shortest_path = reconstruct_path(start, end, prev)

    next_cell = shortest_path[1]

    if next_cell[0] - start[0] == 1:
        # Pseudo-down
        move = 1

    if next_cell[0] - start[0] == -1:
        # Pseudo-up
        move = 2

    if next_cell[1] - start[1] == 1:
        # Pseudo-right
        move = 3

    if next_cell[1] - start[1] == -1:
        # Pseudo-right
        move = 4

    distance = len(shortest_path)

    return (move, distance, shortest_path)


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

    features = state2position_features_cross(game_state=new_field,
                                             agent_position=self_position)

    # layers = 1
    #
    # features = state2position_features_rings(game_state=new_field,
    #                                          agent_position=self_position,
    #                                          layers=layers)

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

        # ADD FEATURES: closest coin stats

        graph = array2graph(new_field)

        for idx, coin in enumerate(coins):

            # Only calculate coins that are accessible
            if new_field[coin[0]][coin[1]] == 0:
                continue

            # Skip coins that spawn/are at agent location
            if self_position == tuple(coin):
                continue
            temp_coin_stats = get_distance_and_move(start=self_position,
                                                    end=tuple(coin),
                                                    graph=graph,
                                                    nr_nodes=15 * 15)

            if idx == 0:

                closest_coin_stats = temp_coin_stats
                closest_coin = coin

            else:
                try:
                    if temp_coin_stats[1] < closest_coin_stats[1]:
                        closest_coin_stats = temp_coin_stats
                        closest_coin = coin
                except UnboundLocalError:
                    closest_coin_stats = temp_coin_stats
                    closest_coin = coin

        x = list("{0:04b}".format(closest_coin[0]))
        x = [int(i) for i in x]
        y = list("{0:04b}".format(closest_coin[1]))
        y = [int(i) for i in y]

        nearest_coin_bin = [*x, *y]
        features.extend(closest_coin)

    else:
        features.extend([0] * 2)

    # ADD FEATURE: Agent's position in binary
    x = list("{0:04b}".format(self_position[0]))
    x = [int(i) for i in x]
    y = list("{0:04b}".format(self_position[1]))
    y = [int(i) for i in y]

    self_position_bin = [*x, *y]
    features.extend(self_position_bin)
    """
    Probably to be deleted bc we already have this...
    #Life-saving features, e.g. whether or not the agent is in the path of a bomb which is about to explode.
    layers = 3
    bomb_area = []
    if len(game_state['bombs']) > 0:
        for x in range(game_state['bombs'][0][0][0] - layers, game_state['bombs'][0][0][0] + layers + 1):
            bomb_area.append([x,game_state['bombs'][0][0][1]])

        for y in range(game_state['bombs'][0][0][1]- layers, game_state['bombs'][0][0][1] + layers + 1):
            bomb_area.append([game_state['bombs'][0][0][0], y])


    if list(self_position) in bomb_area:
        features.extend([1])
    else:
        features.extend([0])
    """

    try:
        # ADD FEATURE: Distance of closest coin to agent
        features.append(closest_coin_stats[1])

        # ADD FEATURE: Move closer to closest coin
        features.append(closest_coin_stats[0])
    except UnboundLocalError:
        features.extend([0] * 2)

    # ADD FEATURE: Dropping bomb possible or not
    features.extend([game_state['self'][2]])
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
