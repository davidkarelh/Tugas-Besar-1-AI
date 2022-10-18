from array import array
from copy import deepcopy
from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import numpy as np
import random

number_of_dots = 4

class RandomBot(Bot):

    def get_action(self, state: GameState) -> GameAction:
        possible_action = self.get_all_possible_action(state)

        max_value = -1
        actionIdx = -1
        for i, action in enumerate(possible_action):
            neighbor_value = self.get_heuristic_value(state, action)
            if max_value < neighbor_value:
                 max_value = neighbor_value
                 actionIdx = i
            elif max_value == neighbor_value:
                if random.random() < 0.5:
                    max_value = neighbor_value
                    actionIdx = i

        return possible_action[actionIdx]

    def get_all_possible_action(self,state: GameState) -> array:
        possible_action = []

        for y, row in enumerate(state.row_status):
            for x, r in enumerate(row):
                if (not r): 
                    possible_action.append(GameAction("row", (x, y)))

        for y, col in enumerate(state.col_status):
            for x, c in enumerate(col):
                if (not c): 
                    possible_action.append(GameAction("col", (x, y)))

        return possible_action

    def get_result(self, state: GameState, action: GameAction) -> GameState:
        x = action.position[0]
        y = action.position[1]
        type = action.action_type
        val = 1
        playerModifier = 1

        newState = deepcopy(state)

        if newState.player1_turn:
            playerModifier = -1

        if y < (number_of_dots-1) and x < (number_of_dots-1):
            newState.board_status[y][x] = (abs(newState.board_status[y][x]) + val) * playerModifier

        if type == 'row':
            newState.row_status[y][x] = 1
            if y >= 1:
                newState.board_status[y-1][x] = (abs(newState.board_status[y-1][x]) + val) * playerModifier

        elif type == 'col':
            newState.col_status[y][x] = 1
            if x >= 1:
                newState.board_status[y][x-1] = (abs(newState.board_status[y][x-1]) + val) * playerModifier

        return newState

    def get_utility_value(self, state: GameState) -> int:
        if state.player1_turn:
            return np.count_nonzero(state.board_status == -4) * 4 - np.count_nonzero(abs(state.board_status) == 3)
        else:
            return np.count_nonzero(state.board_status == 4) * 4 - np.count_nonzero(abs(state.board_status) == 3)

    def get_heuristic_value(self, state: GameState, action: GameAction) -> int:
        current_value = self.get_utility_value(state)

        neighbor_state = self.get_result(state, action)
        neighbor_value = self.get_utility_value(neighbor_state)

        if neighbor_value <= current_value:
             return neighbor_value
        
        possible_actions = self.get_all_possible_action(neighbor_state)

        for action in possible_actions:
            neighbor_value = max(self.get_heuristic_value(neighbor_state, action), neighbor_value)

        return neighbor_value