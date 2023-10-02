from collections import namedtuple, deque
import pickle
import numpy as np
import events as e
from .callbacks import state_to_features
from typing import List
from settings import BOMB_POWER, COLS
from collections import namedtuple, deque
from .dqnmodel import ReplayMemory,DQN
import random
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ast
import re
import os

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 5
RECORD_ENEMY_TRANSITIONS = 1.0
PLACEHOLDER_EVENT = "PLACEHOLDER"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jsonDecoder(Dict):
    return namedtuple('Transition', Dict.keys())(*Dict.values())

def back2nparray(text):

    if(text is None):
        return torch.zeros(25)

    #print("1111111:",text)
    # , -> " "
    #text = text.replace(",", " ")
    #print("222222:", text)
    # delete \n
    text = text.replace('\n', '')
    #print("333333:", text)
    # add ','
    xs = re.sub('\s+', ',', text)
    if (xs[1]==','):
        xs="["+xs[2:]
    else:
        xs = "[" + xs[1:]
    #print("44444:", xs)
    # invert into numpy.array
    a = np.array(ast.literal_eval(xs))

    return torch.tensor(a)

def convert_action(action):
    #'UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'
    if(action=='UP'):
        return torch.tensor([1])
    elif(action=='RIGHT'):
        return torch.tensor([2])
    elif (action == 'DOWN'):
        return torch.tensor([3])
    elif (action == 'LEFT'):
        return torch.tensor([4])
    elif (action == 'WAIT'):
        return torch.tensor([5])
    elif (action == 'BOMB'):
        return torch.tensor([6])
    else:
        return torch.tensor([0])


def setup_training(self):
    self.prev_bombs = deque(maxlen=2)  # First entry = empty list to avoid first round problem
    self.prev_bombs.append([])

    # print("-----train-----setup------")
    # print(self)

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) #maintain a replay buffer (a deque of transitions) to store experiences.
    self.learning_rate = 0.01
    self.discount_factor = 0.95
    self.epsilon = 1.0
    self.epsilon_decay = 0.99
    self.min_epsilon = 0.05
    self.gamma = 0.95
    self.alpha = 0.1

    #####################################

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    self.BATCH_SIZE = 128
    self.GAMMA = 0.99
    self.EPS_START = 0.9
    self.EPS_END = 0.05
    self.EPS_DECAY = 1000
    self.TAU = 0.005
    self.LR = 1e-4

    n_actions = len(ACTIONS)

    self.n_observations = 25  # 3*9+2 game state /feature


    ########load state_dict
    self.policy_net = DQN(self.n_observations, n_actions).to(device)
    self.target_net = DQN(self.n_observations, n_actions).to(device)

    if not os.path.isfile("policy_net_state_dict.pth"):
        pass
    else:
        policy_state_dict = torch.load('policy_net_state_dict.pth')
        self.policy_net.load_state_dict(policy_state_dict)

    if not os.path.isfile("target_net_state_dict.pth"):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    else:
        target_state_dict = torch.load('target_net_state_dict.pth')
        self.target_net.load_state_dict(target_state_dict)


    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
    self.memory = ReplayMemory(10000)
    #print(len(self.memory))
    self.steps_done = 0
    self.episode_durations = []


    ########################
    with open('../../datasets_baseline_all_feature.json', 'r', encoding='utf8') as fp:
        json_data = json.load(fp)#, object_hook=jsonDecoder)

    #print(json_data)


    for i in json_data:
        action=convert_action(json_data[i][1])
        #print(json_data[i])
        self.memory.push(back2nparray(json_data[i][0]),action,back2nparray(json_data[i][2]),torch.tensor([int(json_data[i][3])]))
    #back2nparray
    #state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward

    pretrain(self)


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

def pretrain(self):
    num_round=1000
    for i in range(num_round):
        optimize_model(self)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                        1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    #save state_dict
    #torch.save(self.policy_net.state_dict(), 'policy_net_state_dict.pth')
    #torch.save(self.target_net.state_dict(), 'target_net_state_dict.pth')



def update_q_values(self, gamma):
    #sample batches from the replay buffer and update Q-values based on the Bellman equation.
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

    with open("my-saved-qtable.pkl", "wb") as file:
        pickle.dump(self.q_table, file)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    '''
    # Calculate reward based on events
    reward = reward_from_events(self, events)
    # add transitions to the replay buffer (store the state, action, next state, and reward)
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))
    # Gradually decrease epsilon
    self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
    '''

    ##############################################
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

    # print("-----train-----act------")
    # print(self)

    reward = reward_from_events( events)




    self.memory.push(torch.tensor(state_to_features(old_game_state,self.prev_bombs[0])), convert_action(self_action), torch.tensor(state_to_features(new_game_state,self.prev_bombs[0])), torch.tensor([int(reward)]))

    # Perform one step of the optimization (on the policy network)
    optimize_model(self)

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
    self.target_net.load_state_dict(target_net_state_dict)

    #####save the model
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #update_q_values(self, self.gamma)
    reward = reward_from_events(events)
    self.transitions.append(Transition(torch.tensor(state_to_features(last_game_state)), last_action, None, torch.tensor([int(reward)])))

    torch.save(self.policy_net.state_dict(), 'policy_net_state_dict.pth')
    torch.save(self.target_net.state_dict(), 'target_net_state_dict.pth')

def reward_from_events( events: List[str]) -> float:
    game_rewards = {
        e.COIN_COLLECTED: 200.0,
        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: 0,
        e.SURVIVED_ROUND: 0,
        e.COIN_FOUND: 0,
        e.GOT_KILLED: 0,
        e.CRATE_DESTROYED: 0,
        #PLACEHOLDER_EVENT: 0,
        e.INVALID_ACTION: -50.0,
        #MOVE_CLOSER_TO_COIN: 100.0,
        #MOVE_AWAY_FROM_COIN: -100.0,
        e.WAITED: 0,
        #GOT_STUCK: -100.0
    }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def optimize_model(self):
    if len(self.memory)<self.BATCH_SIZE:
        print("noooo")
        return
    #print("yessssss")

    transitions = self.memory.sample(self.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    #print("batch",batch)


    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    tempppp=[s for s in batch.next_state if s is not None]
    temp_states=torch.cat(tempppp)
    #print(temp_states.shape)
    # print("1111",temp_states.shape[0])
    # print("2222",self.BATCH_SIZE*self.n_observations)
    #print(tempppp)
    if (temp_states.shape[0] < self.BATCH_SIZE*self.n_observations):
        non_final_next_states=torch.cat(tempppp+[tempppp[-1]]*(self.BATCH_SIZE*self.n_observations-temp_states.shape[0]))[:self.BATCH_SIZE*self.n_observations]
    else:
        non_final_next_states = temp_states[:self.BATCH_SIZE*self.n_observations]
    # print("333333",non_final_next_states.shape)


    if(len(batch.state)<self.BATCH_SIZE*self.n_observations):
        state_batch=torch.cat(batch.state+tuple([batch.state[-1]]*(self.BATCH_SIZE*self.n_observations-len(batch.state))))[:self.BATCH_SIZE*self.n_observations]
    else:
        state_batch = torch.cat(batch.state)[:self.BATCH_SIZE*self.n_observations]
    #print(batch.action)
    #
    # if (len(batch.action) < self.BATCH_SIZE * self.n_observations):
    #     action_batch = torch.cat(
    #         batch.action + tuple([batch.action[-1]] * (self.BATCH_SIZE * self.n_observations - len(batch.action))))[
    #                   :self.BATCH_SIZE * self.n_observations]
    # else:
    #     action_batch = torch.cat(batch.action)[:self.BATCH_SIZE*self.n_observations]

    # if (len(batch.reward) < self.BATCH_SIZE * self.n_observations):
    #     reward_batch = torch.cat(
    #         batch.reward + tuple([batch.reward[-1]] * (self.BATCH_SIZE * self.n_observations - len(batch.reward))))[
    #                   :self.BATCH_SIZE * self.n_observations]
    # else:
    #     reward_batch = torch.cat(batch.reward)[:self.BATCH_SIZE*self.n_observations]
    #state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)

    #print(state_batch)

    state_batch=state_batch.reshape(self.BATCH_SIZE,self.n_observations)
    action_batch = action_batch.reshape(1,action_batch.shape[0])
    #print("action_batch", action_batch.shape)

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
    non_final_next_states=non_final_next_states.reshape(self.BATCH_SIZE,self.n_observations)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

    print(state_action_values.shape)
    print(expected_state_action_values.unsqueeze(0).shape)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.requires_grad_(True)
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()
