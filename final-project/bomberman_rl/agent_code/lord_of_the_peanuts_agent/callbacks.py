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


def state_to_features_ori(game_state: dict) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    self_position = game_state["self"][3]
    new_field = field2bomb(game_state)
    new_field = field2coin(game_state, new_field)

    features = []
    for x in range(new_field.shape[0]):
        for y in range(new_field.shape[1]):
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


def state_to_features(game_state: dict) -> np.array:

    if game_state is None:
        return None

    #reset the field to accelerate the algorithm
    field = game_state["field"]
    self_position = game_state["self"][3]
    new_field = field2bomb(game_state)
    new_field = field2coin(game_state, new_field)

    # !!think more about the features, such as vector to all coins

    # reduce the feature
    # layers=0 means that only consider the information of actual field
    layers=1
    features = []
    #max(bomb[0][0]-power,0):min(bomb[0][0]+power+1
    for x in range(self_position[0]-layers,self_position[0]+layers+1):
        for y in range(self_position[1]-layers,self_position[1]+layers+1):
            #-1 means the border of the map
            #-2 means expetional situation
            #we wont have -2
            #we do this because we want feature have a fixed length
            #print(x,y)
            if (x<0 or x>=new_field.shape[0] or y<0 or y>=new_field.shape[1]):
                cell=-2
            else:
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
    #print(features)
    return np.array(features)


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

