from xmlrpc.client import boolean
from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import random
import numpy as np
from copy import deepcopy
from typing import Tuple

class KyleBot(Bot):
    def __init__(self):
        self.rows = None
        self.cols = None
        self.turn = None
        self.dots = None

    def get_action(self, state: GameState) -> GameAction:
        if self.rows is None:  #Also implies cols is None, which not yet know board size
            self.rows = {
                "y": state.row_status.shape[0],
                "x": state.row_status.shape[1]
            }
            self.cols = {
                "y": state.col_status.shape[0],
                "x": state.col_status.shape[1]
            }
            self.board_size = {
                "y": state.board_status.shape[0],
                "x": state.board_status.shape[1]
            }
            self.dots = max(self.rows['y'], self.rows['x'])
            self.turn = -1 if state.player1_turn else 1
        # Expand and check moves, while in alphabeta we dont need to track moves
        expands = self.get_actions_first(state)
        states = expands[0]
        moves = expands[1]
        max_score = -np.inf
        n = len(states)
        for i in range(n):
            state = states[i]
            if self.test_terminal(state[0]):
                score = self.utility(state[0])
            else:
                if state[1]:
                    score = max(max_score, self.alphabeta(state[0], -np.inf, np.inf, True))
                else:
                    score = max(max_score, self.alphabeta(state[0], -np.inf, np.inf, False))
            if score > max_score:
                max_score = score
                best_move = moves[i]
        print(max_score)
        return best_move
        # Return the move with the highest score

        
    def alphabeta(self, state: GameState, alpha: float, beta: float, maximizingplayer: bool) -> GameAction:
        '''
        Returns the best action for the current player on the board.
        '''
        if self.test_terminal(state):
            return self.utility(state.board_status)
        if maximizingplayer:
            value = -np.inf
            for action in self.get_actions(state, maximizingplayer):
                new_state = action[0]
                claimed = action[1]
                if claimed:
                    value = max(value, self.alphabeta(new_state, alpha, beta, True))
                else: 
                    value = max(value, self.alphabeta(new_state, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            for action in self.get_actions(state, maximizingplayer):
                new_state = action[0]
                claimed = action[1]
                if claimed:
                    value = min(value, self.alphabeta(new_state, alpha, beta, False))
                else: 
                    value = min(value, self.alphabeta(new_state, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def get_actions(self, state: GameState, maximizing : bool) -> list[Tuple[GameState,bool]]:
        '''
        Returns a list of all possible actions.
        '''
        actions = []
        for i in range(self.rows['y']):
            for j in range(self.rows['x']):
                if state.row_status[i][j] == 0:
                    actions.append(self.move(state, GameAction("row", (j, i)), maximizing))
        for i in range(self.cols['y']):
            for j in range(self.cols['x']):
                if state.col_status[i][j] == 0:
                    actions.append(self.move(state, GameAction("col", (j, i)), maximizing))
        return actions
    
    def get_actions_first(self, state: GameState) -> Tuple[list[Tuple[GameState,bool]], list[GameAction]]:
        '''
        Returns a list of all possible actions on first move.
        '''
        actions = []
        moves = []
        for i in range(self.rows['y']):
            for j in range(self.rows['x']):
                if state.row_status[i][j] == 0:
                    actions.append(self.move(state, GameAction("row", (j, i)), True))
                    moves.append(GameAction("row", (j, i)))
        for i in range(self.cols['y']):
            for j in range(self.cols['x']):
                if state.col_status[i][j] == 0:
                    actions.append(self.move(state, GameAction("col", (j, i)), True))
                    moves.append(GameAction("col", (j, i)))
        return actions, moves

    def move(self, state: GameState, action: GameAction, maximizing: bool) -> Tuple[GameState, bool]:
        '''
        Returns the board that results from making move (i, j) on the board.
        '''
        new_state = deepcopy(state)
        y = action.position[1]
        x = action.position[0]
        scored = False
        playerModifier = self.turn if maximizing else -self.turn
        if y < (self.dots-1) and x < (self.dots-1):
            new_state.board_status[y][x] = (abs(new_state.board_status[y][x]) + 1) * playerModifier
            if abs(new_state.board_status[y][x]) == 4:
                scored = True
        if action.action_type == 'row':
            new_state.row_status[y][x] = 1
            if y >= 1:
                new_state.board_status[y-1][x] = (abs(new_state.board_status[y-1][x]) + 1) * playerModifier
                if abs(new_state.board_status[y-1][x]) == 4:
                    scored = True
        elif action.action_type == 'col':
            new_state.col_status[y][x] = 1
            if x >= 1:
                new_state.board_status[y][x-1] = (abs(new_state.board_status[y][x-1]) + 1) * playerModifier
                if abs(new_state.board_status[y][x-1]) == 4:
                    scored = True
        return new_state, scored

    def utility(self, board_status: np.ndarray) -> int:
        '''
        Returns the utility of the board.
        '''
        self_score = 4 * self.turn
        opponent_score = -4 * self.turn
        score = len(np.argwhere(board_status == self_score))
        score -= len(np.argwhere(board_status == opponent_score))
        return score
        

    def test_terminal(self, state: GameState) -> bool:
        '''
        Test if the game is over.
        TODO: Implement heuristic early termination
        '''
        return np.all(state.row_status == 1) and np.all(state.col_status == 1)