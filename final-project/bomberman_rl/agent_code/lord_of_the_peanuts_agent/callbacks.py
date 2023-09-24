import os
import pickle
import numpy as np
import math
import logging
from settings import BOMB_POWER, COLS
from datetime import datetime

from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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

    self.prev_bombs = deque(maxlen=2)  # First entry = empty list to avoid first round problem
    self.prev_bombs.append([])

def act(self, game_state: dict) -> str:

    # Save current round bomb positions for next round
    self.prev_bombs.append([bomb for bomb in game_state['bombs'] if bomb[1] == 0])
    # curr_active_bombs = []
    # for bomb in game_state['bombs']:
    #     if bomb[1] == 0:
    #         curr_active_bombs.append(bomb)

    if game_state is None:
        return np.random.choice(ACTIONS)

    state = state_to_features(self, game_state, self.prev_bombs[0])[0]
    if self.train and np.random.rand() < self.epsilon:
        return np.random.choice(ACTIONS)

    q_values = self.q_table.get(tuple(state), {a: 0.0 for a in ACTIONS})
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]

    return np.random.choice(best_actions)


def state2position_features_cross(game_state, agent_position) -> list:
    """
    Convert the game_state to an array of features describing the agent's surroundings.
    Only take the tiles above, below, left and right from the agent.
    Input:
    - game_state: np.array
    - agent_position: list
    Output:
    - features: list (len=12=4*3)
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
    centre_cell = int(len(features) / 2)
    del features[centre_cell - 1: centre_cell + 2]

    return features


# TO-DO:  make function for when we test hyperparameters
#     position = np.array(self_position, dtype="int")
#     nearest_coin = coins[np.argmin(np.sum(np.abs(coins - position), axis=1))]


# def array2graph(array, self_pos: tuple, nogo: list = [-1, 1, 2]) -> dict:
def array2graph(array, nogo: list = [-1, 1, 2]) -> dict:

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

    if not prev:
        return (-1, -1, ())

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

    distance = len(shortest_path) - 1

    return (move, distance, shortest_path)


def movement_away_bomb(agent_pos: tuple, bomb_pos: tuple) -> int:
    """
    Returns the direction the agent should take to move away from a bomb. Based on angles.
    """
    assert not agent_pos == bomb_pos

    # Calculate angles between agent and bombs
    angle = math.atan2(bomb_pos[1] - agent_pos[1], bomb_pos[0] - agent_pos[0])
    if angle < 0:
        angle = angle + (2 * math.pi)

    # Correct for cartesian -> array coordinates
    angle = angle - (math.pi / 2)

    # Calculate the opposite direction (180 degrees away)
    opposite_direction = (angle - math.pi) % (2 * math.pi)

    if 0 <= opposite_direction < math.pi / 4:
        return 1  # RIGHT
    elif math.pi / 4 <= opposite_direction < 3 * math.pi / 4:
        return 2  # UP
    elif 3 * math.pi / 4 <= opposite_direction < 5 * math.pi / 4:
        return 3  # LEFT
    elif 5 * math.pi / 4 <= opposite_direction < 7 * math.pi / 4:
        return 4  # DOWN
    elif 7 * math.pi / 4 <= opposite_direction < 2 * math.pi:
        return 1  # RIGHT
    else:
        assert False, f"Opposite direction{opposite_direction}, agent: {agent_pos}, bomb: {bomb_pos}"


def state_to_features(self, game_state: dict, prev_bombs: list) -> list:
    if game_state is None:
        return None

    self_position = game_state["self"][3]
    new_field = field2bomb(game_state)

    # Update field with with previous round's explosion
    if prev_bombs:

        bomb_mask = np.zeros([COLS,COLS])

        for bomb in prev_bombs:
            if bomb[1] == 0:
                bomb_mask[max(bomb[0][0] - BOMB_POWER, 0):min(bomb[0][0] + BOMB_POWER + 1, COLS), bomb[0][1]] = 1
                bomb_mask[bomb[0][0]][max(bomb[0][1] - BOMB_POWER, 0):min(bomb[0][1] + BOMB_POWER + 1, COLS)] = 1

        updated_mask = (new_field == 0) & np.array(bomb_mask, dtype=bool)
        new_field = np.where(updated_mask, 2, new_field)

    # Update field with coin locations
    new_field = field2coin(game_state, new_field)
    features = state2position_features_cross(game_state=new_field,
                                             agent_position=self_position)

    coins = game_state['coins']

    if len(coins) > 0 and (coins != [self_position]):

        # ADD FEATURES: closest coin stats

        graph = array2graph(new_field)

        exception_counter = 0

        for idx, coin in enumerate(coins):

            # Skip coins that spawn/are at agent location
            if self_position == coin:
                exception_counter += 1
                continue

            if new_field[self_position] == 2:
                exception_counter += 1
                continue

            temp_coin_stats = get_distance_and_move(start=self_position,
                                                    end=coin,
                                                    graph=graph,
                                                    nr_nodes=15 * 15)
            if not temp_coin_stats[2]:
                exception_counter += 1
                continue

            if idx == 0 or (idx == 1 and coins[0] == self_position) or (idx == 1 and new_field[coins[0]] == 2) or (idx == exception_counter):#closest_coin_stats[1] == -1:

                closest_coin_stats = temp_coin_stats
                closest_coin = coin

            else:
                if temp_coin_stats[1] < closest_coin_stats[1]:
                    closest_coin_stats = temp_coin_stats
                    closest_coin = coin

    # ADD FEATURE: Agent's position in binary
    features.extend(self_position)

    try:
        # ADD FEATURE: Distance of closest coin to agent
        coin_distance = closest_coin_stats[1]

        if coin_distance == 0:
            features.append(0)
        else:
            features.append((coin_distance - 1) // 4)  # 7 bins

        # ADD FEATURE: Move closer to closest coin
        features.append(closest_coin_stats[0])

    except UnboundLocalError:
        features.extend([0] * 2)
        coin_distance = 0

    # ADD FEATURE: loaded (bomb)
    features.append(int(game_state['self'][2]))


    # Closest bomb
    if game_state['bombs']: # and not (len(game_state['bombs']) == 1 and game_state['bombs'][0][0] == self_position):

        for idx, bomb in enumerate(game_state['bombs']):

            # Skip coins that spawn/are at agent location
            # if self_position == bomb[0]:
            #     continue

            # Manhattan distance
            bomb_pos = bomb[0]

            manhattan_dist = abs(bomb_pos[0] - self_position[0]) + abs(bomb_pos[1] - self_position[1])

            if idx == 0: # or (game_state['bombs'][0][0] == self_position and idx == 1):

                closest_bomb = bomb_pos
                closest_bomb_dist = manhattan_dist

            else:

                if manhattan_dist < closest_bomb_dist:

                    closest_bomb = bomb_pos
                    closest_bomb_dist = manhattan_dist

        # ADD FEATURE: distance to closest bomb
        if closest_bomb_dist == 0:
            features.append(0)
        elif closest_bomb_dist == 1:
            features.append(1)
        elif closest_bomb_dist == 2:
            features.append(2)
        elif closest_bomb_dist == 3:
            features.append(3)
        elif ( 3 < closest_bomb_dist and closest_bomb_dist <= 6):
            features.append(4)
        elif (6 < closest_bomb_dist and closest_bomb_dist <= 10):
            features.append(5)
        elif (10 < closest_bomb_dist and closest_bomb_dist <= 18):
            features.append(6)
        else:
            features.append(7)

        # ADD FEATURE: move away of closest bomb
        if closest_bomb != self_position:
            away_direction = movement_away_bomb(self_position, closest_bomb)
        else:
            away_direction = 0

        features.append(away_direction)

    else:
        features.extend([0, -1]) # No bombs, so distance is 0
        closest_bomb_dist = 0

    # ADD FEATURE: Opponent positions in bins
    if game_state["others"]:

        for idx, opponent in enumerate(game_state["others"]):

            # Manhattan distance
            distance_to_opponent = np.abs(self_position[0] - opponent[3][0]) + np.abs(self_position[1] - opponent[3][1])

            if idx == 0:

                closest_agent = opponent
                closest_opponent_dist = distance_to_opponent

            else:
                if distance_to_opponent < closest_opponent_dist:
                    closest_agent = opponent
                    closest_opponent_dist = distance_to_opponent


            distance_to_opponent_bins = (closest_opponent_dist - 1) // 4 # 7 bins
            features.append(distance_to_opponent_bins)

    else:
        features.append(0)
        closest_opponent_dist = 0


    return [np.array(features), coin_distance, closest_bomb_dist, closest_opponent_dist]



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
