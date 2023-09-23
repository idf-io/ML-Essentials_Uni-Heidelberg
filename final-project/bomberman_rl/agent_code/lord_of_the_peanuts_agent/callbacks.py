import os
import pickle
import numpy as np
import logging
from settings import BOMB_POWER, COLS
from collections import namedtuple, deque
from .dqnmodel import ReplayMemory,DQN
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
steps_done = 0
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor2str(action):
    if (action == torch.tensor([1])):
        return 'UP'
    elif (action == torch.tensor([2])):
        return 'RIGHT'
    elif (action == torch.tensor([3])):
        return 'DOWN'
    elif (action == torch.tensor([4])):
        return 'LEFT'
    elif (action == torch.tensor([5])):
        return 'WAIT'
    elif (action == torch.tensor([6])):
        return 'BOMB'
    else:
        return None

def setup(self):


    self.epsilon = 1.0
    self.learning_rate = 0.1  # Learning rate for Q-learning
    self.discount_factor = 0.9  # Discount factor for future rewards
    if self.train and  not os.path.isfile("my-saved-qtable.pkl"):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {} # Initialize Q-table
    else:
        #already saved some parameters
        #the result of q-learning is q-table
        self.logger.info("Loading Q-table from saved state.")
        with open("my-saved-qtable.pkl", "rb") as file:
            self.q_table = pickle.load(file)





def act(self, game_state: dict) -> str:

    if game_state is None:
        return np.random.choice(ACTIONS)
    '''
    state = state_to_features(game_state)
    if self.train and np.random.rand() < self.epsilon:
        return np.random.choice(ACTIONS)

    q_values = self.q_table.get(tuple(state), {a: 0.0 for a in ACTIONS})
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]
    
    '''
    ######################################
    state_numpy=state_to_features(game_state)
    #invert into torch tensor
    state = torch.tensor(state_numpy, dtype=torch.float32, device=device).unsqueeze(0)

    action = select_action(self,state)

    if (action not in ACTIONS):
        action=None

    return action
    #return np.random.choice(best_actions)


def select_action(self,state):
    # will select an action accordingly to an epsilon greedy policy.
    # Simply put, we’ll sometimes use our model for choosing the action,
    # and sometimes we’ll just sample one uniformly. The probability of choosing
    # a random action will start at and will decay exponentially towards .
    # controls the rate of the decay.EPS_STARTEPS_ENDEPS_DECAY
    sample = random.random()
    eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * self.steps_done / self.EPS_DECAY)
    self.steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1)[1].view(1, 1)
    else:
        #return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return np.random.choice(ACTIONS)
    '''
    if self.train and np.random.rand() < self.epsilon:
        return np.random.choice(ACTIONS)

    q_values = self.q_table.get(tuple(state), {a: 0.0 for a in ACTIONS})
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]
    return np.random.choice(best_actions)
    '''


##################################################
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

    distance = len(shortest_path) - 1

    return (move, distance, shortest_path)


def state_to_features(game_state: dict) -> list:
    if game_state is None:
        return None

    self_position = game_state["self"][3]
    new_field = field2bomb(game_state)
    new_field = field2coin(game_state, new_field)
    features = state2position_features_cross(game_state=new_field,
                                             agent_position=self_position)

    coins = game_state['coins']





    # ADD FEATURE: Agent's position in binary
    features.extend(self_position)



    # ADD FEATURE: loaded (bomb)
    features.append(int(game_state['self'][2]))

    # ADD FEATURE: distance to closest bomb
    bomb_distances = []  # To store distances to bombs
    for bomb in game_state["bombs"]:
        bomb_pos = bomb[0]
        manhattan_dist = abs(bomb_pos[0] - self_position[0]) + abs(bomb_pos[1] - self_position[1])
        bomb_distances.append(manhattan_dist)

    # Use the minimum Manhattan distance to the nearest bomb
    if bomb_distances:
        nearest_bomb_dist = min(bomb_distances)
        bomb_distance = nearest_bomb_dist
        features.append(nearest_bomb_dist)
    else:
        bomb_distance = 0
        features.append(0)  # No bombs, so distance is 0


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