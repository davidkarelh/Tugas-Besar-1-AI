from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import numpy as np
from copy import deepcopy
from typing import Tuple
from time import time

class TEntry:
    def __init__(self, value: int, depth: int, type: str):
        self.value = value
        self.depth = depth
        self.type = type
class KyleBot(Bot):
    def __init__(self, max_depth: int = 7, max_second: int = 5):
        self.rows = None
        self.cols = None
        self.turn = None
        self.dots = None
        self.ttable = {}
        self.row_zobrist = None
        self.col_zobrist = None
        self.me_zobrist = None
        self.opp_zobrist = None
        self.playermax_hash = None
        self.playermin_hash = None
        self.change_hash = None
        self.max_depth = max_depth
        self.max_second = max_second

    def get_action(self, state: GameState) -> GameAction:
        start_sec = time()
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

        # Zobrist hashing
        if self.row_zobrist is None:
            self.row_zobrist = np.random.randint(0, 2**32-1, size=(self.rows['y'], self.rows['x']), dtype=np.uint32)
            self.col_zobrist = np.random.randint(0, 2**32-1, size=(self.cols['y'], self.cols['x']), dtype=np.uint32)
            self.playermax_hash = np.random.randint(0, 2**32-1, dtype=np.uint32)
            self.playermin_hash = np.random.randint(0, 2**32-1, dtype=np.uint32)
            self.me_zobrist = np.random.randint(0, 2**32-1, size=(self.dots-1, self.dots-1),dtype=np.uint32)
            self.opp_zobrist = np.random.randint(0, 2**32-1, size=(self.dots-1, self.dots-1),dtype=np.uint32)
            self.change = self.playermax_hash ^ self.playermin_hash
        self.zval_init(state)

        # Expand and check moves, while in alphabeta next moves we dont need to track moves
        expands = self.get_actions_moves(state, True)
        states = expands[0]
        moves = expands[1]
        max_score = -np.inf
        self.n = len(states)
        each_state_max = (self.max_second / self.n) - (0.0001/self.n)
        for i in range(self.n):
            self.max_time = start_sec + each_state_max * (i + 1)
            state = states[i]
            move = moves[i]
            self.zval ^= getattr(self, move.action_type + "_zobrist")[move.position[1]][move.position[0]]
            if self.test_terminal(state[0]):
                score = self.utility(state[0])
            else:
                if state[1]:
                    score = max(max_score, self.alphabeta(state[0], -np.inf, np.inf, True, 0))
                else:
                    self.zval ^= self.change
                    score = max(max_score, self.alphabeta(state[0], -np.inf, np.inf, False, 0))
                    self.zval ^= self.change
            if score > max_score:
                max_score = score
                best_move = moves[i]
            self.zval ^= getattr(self, move.action_type + "_zobrist")[move.position[1]][move.position[0]]
        return best_move
        # Return the move with the highest score

    def zval_init(self, state: GameState):
        '''
        Initializes the zobrist hash value.
        '''
        self.zval = 0
        for i in range(self.rows['y']):
            for j in range(self.rows['x']):
                if state.row_status[i][j] != 0:
                    self.zval ^= self.row_zobrist[i][j]
        for i in range(self.cols['y']):
            for j in range(self.cols['x']):
                if state.col_status[i][j] != 0:
                    self.zval ^= self.col_zobrist[i][j]
        self.zval ^= self.playermax_hash
        
    def alphabeta(self, state: GameState, alpha: float, beta: float, maximizingplayer: bool, depth : int) -> GameAction:
        '''
        Returns the best action for the current player on the board.
        '''
        alphaorig = alpha

        # Cache first
        cached = self.ttable.get(self.zval, None)
        if cached is not None and cached.depth >= depth:
            if cached.type == 'exact':
                return cached.value
            elif cached.type == 'lower':
                alpha = max(alpha, cached.value)
            else:
                beta = min(beta, cached.value)
            
            if alpha >= beta:
                return cached.value

        base = self.zval
        # Terminal test
        if self.test_terminal(state):
            return self.utility(state.board_status)
        
        if depth >= self.max_depth:
            if time() >= self.max_time-(0.1/self.n):
                return self.utility(state.board_status)
            else:
                self.max_depth += 1

        if maximizingplayer:
            value = -np.inf
            actions, moves = self.get_actions_moves(state, maximizingplayer)
            for i in range(len(actions)):
                if time() > self.max_time:
                    return value
                move = moves[i]
                action = actions[i]
                new_state = action[0]
                claimed = action[1]
                self.zval ^= getattr(self, move.action_type + "_zobrist")[move.position[1]][move.position[0]]
                if claimed:
                    z = self.captured_zval(move, new_state)
                    self.zval ^= z
                    value = max(value, self.alphabeta(new_state, alpha, beta, maximizingplayer, depth+1))
                    
                else:
                    self.zval ^= self.change
                    value = max(value, self.alphabeta(new_state, alpha, beta, not maximizingplayer, depth+1))
                    self.zval ^= self.change
                alpha = max(alpha, value)
                self.zval ^= getattr(self, move.action_type + "_zobrist")[move.position[1]][move.position[0]]
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            actions, moves = self.get_actions_moves(state, maximizingplayer)
            for i in range(len(actions)):
                if time() > self.max_time:
                    return value
                move = moves[i]
                self.zval ^= getattr(self, move.action_type + "_zobrist")[move.position[1]][move.position[0]]
                action = actions[i]
                new_state = action[0]
                claimed = action[1]
                if claimed:
                    z = self.captured_zval(move, new_state)
                    self.zval ^= z
                    value = min(value, self.alphabeta(new_state, alpha, beta, maximizingplayer, depth+1))
                    self.zval ^= z
                else:
                    self.zval ^= self.change
                    value = min(value, self.alphabeta(new_state, alpha, beta, not maximizingplayer, depth+1))
                    self.zval ^= self.change
                beta = min(beta, value)
                self.zval ^= getattr(self, move.action_type + "_zobrist")[move.position[1]][move.position[0]]
                if beta <= alpha:
                    break
            if value <= alphaorig:
                self.ttable[base] = TEntry(value, depth, "upper")
            elif value >= beta:
                self.ttable[base] = TEntry(value, depth, "lower")
            else:
                self.ttable[base] = TEntry(value, depth, "exact")
            return value
    
    def get_actions_moves(self, state: GameState, maximizing) -> Tuple[list[Tuple[GameState,bool]], list[GameAction]]:
        '''
        Returns a list of all possible actions on first move.
        '''
        actions = []
        moves = []
        for i in range(self.rows['y']):
            for j in range(self.rows['x']):
                if state.row_status[i][j] == 0:
                    actions.append(self.move(state, GameAction("row", (j, i)), maximizing))
                    moves.append(GameAction("row", (j, i)))
        for i in range(self.cols['y']):
            for j in range(self.cols['x']):
                if state.col_status[i][j] == 0:
                    actions.append(self.move(state, GameAction("col", (j, i)), maximizing))
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
        else: # Col
            new_state.col_status[y][x] = 1
            if x >= 1:
                new_state.board_status[y][x-1] = (abs(new_state.board_status[y][x-1]) + 1) * playerModifier
                if abs(new_state.board_status[y][x-1]) == 4:
                    scored = True
        return new_state, scored
    
    def captured_zval(self, move: GameAction, state: GameState) -> int:
        '''
        Updates the zobrist value of the board with occupied squares.
        '''
        tot = 0
        y = move.position[1]
        x = move.position[0]
        
        if y < (self.dots-1) and x < (self.dots-1):
            if state.board_status[y][x] == 4 * self.turn:
                tot ^= self.me_zobrist[y][x]
            elif state.board_status[y][x] == 4 * -self.turn:
                tot ^= self.opp_zobrist[y][x]
        if move.action_type == 'row':
            if state.board_status[y-1][x] == 4 * self.turn:
                tot ^= self.me_zobrist[y-1][x]
            elif state.board_status[y-1][x] == 4 * -self.turn:
                tot ^= self.opp_zobrist[y-1][x]
        else:
            if state.board_status[y][x-1] == 4 * self.turn:
                tot ^= self.me_zobrist[y][x-1]
            elif state.board_status[y][x-1] == 4 * -self.turn:
                tot ^= self.opp_zobrist[y][x-1]
        return tot
        

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