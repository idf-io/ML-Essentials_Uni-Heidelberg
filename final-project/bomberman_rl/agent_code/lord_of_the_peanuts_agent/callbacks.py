import os
import pickle
import numpy as np
import logging
from settings import BOMB_POWER, COLS
from collections import namedtuple, deque
from .dqnmodel import ReplayMemory,DQN
import gym
import gym_toytext
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
    if self.train or not os.path.isfile("my-saved-qtable.pkl"):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {} # Initialize Q-table
    else:
        #already saved some parameters
        #the result of q-learning is q-table
        self.logger.info("Loading Q-table from saved state.")
        with open("my-saved-qtable.pkl", "rb") as file:
            self.q_table = pickle.load(file)


    #####################################

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    self.BATCH_SIZE = 10
    self.GAMMA = 0.99
    self.EPS_START = 0.9
    self.EPS_END = 0.05
    self.EPS_DECAY = 1000
    self.TAU = 0.005
    self.LR = 1e-4

    n_actions = len(ACTIONS)

    n_observations = 29 #3*9+2

    self.policy_net = DQN(n_observations, n_actions).to(device)
    self.target_net = DQN(n_observations, n_actions).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
    self.memory = ReplayMemory(10000)
    self.steps_done = 0
    self.episode_durations = []

    '''
    # Get the all the lines in file in a list
    action_log = []
    state_log=[]
    with  open("../../test_action.txt", "r") as myfile:
        for line in myfile:
            action_log.append(line.strip())
    with  open("../../test_state.txt", "r") as myfile2:
        for line2 in myfile2:
            state_log.append(line2.strip())

    #print("action_log:",action_log)
    #print("state_log:",state_log)
    '''


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
    '''
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    # Move to the next state
    state = next_state
    '''
    # Perform one step of the optimization (on the policy network)
    optimize_model(self)

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
    self.target_net.load_state_dict(target_net_state_dict)


    return action
    #return np.random.choice(best_actions)


def select_action(self,state):
    # will select an action accordingly to an epsilon greedy policy.
    # Simply put, we’ll sometimes use our model for choosing the action,
    # and sometimes we’ll just sample one uniformly. The probability of choosing
    # a random action will start at and will decay exponentially towards .
    # controls the rate of the decay.EPS_STARTEPS_ENDEPS_DECAY
    global steps_done
    sample = random.random()
    eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * steps_done / self.EPS_DECAY)
    steps_done += 1
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

def optimize_model(self):
    if len(self.memory)<self.BATCH_SIZE:
        print("unexpected!!!!!!!!!!!!!!!")
        return

    transitions = self.memory.sample(self.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()


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

