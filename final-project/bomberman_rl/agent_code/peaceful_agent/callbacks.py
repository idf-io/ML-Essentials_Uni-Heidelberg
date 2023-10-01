import numpy as np


def setup(self):
    np.random.seed()
    # if not self.train:
    #     self.empty_coins_counter = 0


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random, but no bombs.')

    # if not self.train:
    #
    # #
    #     if game_state['step'] == 1:
    #         self.empty_coins_counter = 0
    #
    #     manhd = abs(game_state['coins'][0][0] - game_state['self'][3][0]) + abs(game_state['coins'][0][1] - game_state['self'][3][1])
    #
    #     if len(game_state['coins']) == 1 and manhd == 1:
    #          self.empty_coins_counter += 1
    #
    #     if len(game_state['coins']) == 1 and self.empty_coins_counter == 1:
    #
    #         with open("../../results/win.log", "a") as f:
    #             f.write(str(game_state['step'] - 1))
    #             f.write("\n")


    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
