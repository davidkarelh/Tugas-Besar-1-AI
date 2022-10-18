from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import random
import numpy as np
import math
from typing import NamedTuple
from numpy import ndarray

class TranspositionTableKey(NamedTuple):
    row_status_hash: int
    col_status_hash: int
    board_status_hash: int

class GameStateCopy:
    """
    board_status: int[][]
        For each element, if its absolute element is four, then
        the square has been taken by a player. If element's sign
        is negative, then it has been taken by player 1. Otherwise,
        it has been taken by player 2.
        Access: board_status[y, x]

    row_status: int[][]
        Represent the horizontal line mark status: 1 for marked, 0 for not.
        Access: row_status[y, x]

    col_status: int[][]
        Represent the vertical line mark status: 1 for marked, 0 for not.
        Access: col_status[y, x]
        
    player1_turn: bool
        True if it is player 1 turn, False for player 2.
    """

    board_status: ndarray
    row_status: ndarray
    col_status: ndarray
    player1_turn: bool

class MinimaxDavidBot(Bot):
    number_of_dots = 4
    number_of_nodes = 0
    number_of_cuts = 0
    __transposition_table = {}
    __spiral_move_row_or_column_helper = [0, 1, 0, 1, 1, 1, 0, 0, 1,  1, 0, 0, 0, 0, 0, 1, 1, 1,  0,  0,  0, 1, 1, 1]
    __spiral_move_row_or_column_index =  [4, 6, 7, 5, 1, 2, 5, 8, 10, 9, 6, 3, 0, 1, 2, 3, 7, 11, 11, 10, 9, 0, 4, 8]
    __limit_depth = 0
    __depth_decrease = 4
    __time_to_decrease = True

    def get_action(self, state: GameStateCopy) -> GameAction:
        state_copy = GameStateCopy()
        state_copy.board_status = state.board_status
        state_copy.row_status = state.row_status
        state_copy.col_status = state.col_status
        state_copy.player1_turn = state.player1_turn
        self.number_of_nodes = 0
        self.number_of_cuts = 0

        empty_r_c = self.__get_number_of_empty_row_and_col(state_copy)

        if empty_r_c >= 23:
           self. __limit_depth = 0
           self.__depth_decrease = 5
           self.__time_to_decrease = True
        
        if empty_r_c == 24:
            random_start = random.randint(0, 3)
            line_type = "row"

            position = self.__spiral_move_row_or_column_index[random_start]
            if self.__spiral_move_row_or_column_helper[random_start] == 1:
                line_type = "col"
            
            if line_type == "row":
                y = position // 3
                x = position - (y * 3)
                return GameAction(line_type, (x, y))

            elif line_type == "col" :  
                y = position // 4
                x = position - (y * 4)
                return GameAction(line_type, (x, y))
        
        self.__limit_depth = max(empty_r_c - self.__depth_decrease, 0)

        if self.__time_to_decrease:
            self.__depth_decrease += 1.4
        
        self.__time_to_decrease = not self.__time_to_decrease
        
        return self.__miniMax(state_copy)
    
    def __get_number_of_empty_row_and_col(self, state: GameStateCopy):
        ret = 0
        ret += np.count_nonzero(state.row_status == 0)
        ret += np.count_nonzero(state.col_status == 0)
        return ret

    def update_board_and_check_additional_turn(self, type, position, playerModifier, val, state: GameStateCopy):
        additional_turn = False

        if (type == "row"):
            y = position // 3
            x = position - (y * 3)

        elif (type == "col"):
            y = position // 4
            x = position - (y * 4)
        
        board_status = state.board_status 
        row_status = state.row_status 
        col_status = state.col_status 

        if y < (self.number_of_dots-1) and x < (self.number_of_dots-1):
            board_status[y][x] = (abs(board_status[y][x]) + val) * playerModifier
            if abs(board_status[y][x]) == 4:
                additional_turn = True

        if type == 'row':
            row_status[y][x] = 1 if val == 1 else 0
            if y >= 1:
                board_status[y-1][x] = (abs(board_status[y-1][x]) + val) * playerModifier
                if abs(board_status[y-1][x]) == 4:
                    additional_turn = True

        elif type == 'col':
            col_status[y][x] = 1 if val == 1 else 0
            if x >= 1:
                board_status[y][x-1] = (abs(board_status[y][x-1]) + val) * playerModifier
                if abs(board_status[y][x-1]) == 4:
                    additional_turn = True
                
        return additional_turn

    def __miniMax(self, state: GameStateCopy) -> GameAction:
        
        maxScore = -math.inf
        alpha = -math.inf
        beta = math.inf
        playerModifier = -1 if state.player1_turn else 1
        best_action_type = "row"
        best_action_idx = 0

        hash_state_result = self.__hash_state(state)
        
        exist, transposition_table_result = self.__check_exist_in_transposition_table(state)
        if exist:
            if transposition_table_result[0] >= beta:
                self.number_of_cuts += 1
                return transposition_table_result[0]

            elif transposition_table_result[1] <= alpha:
                self.number_of_cuts += 1
                return transposition_table_result[1]

            alpha = max(alpha, transposition_table_result[0])
            beta = min(beta, transposition_table_result[1])


        row_or_columns, indexes = self.__get_best_moves(state)

        for i, idx in enumerate(indexes):
        
            line_type = "row" if row_or_columns[i] == 0 else "col"
            self.number_of_nodes += 1
            temp = 0

            if self.update_board_and_check_additional_turn(line_type, idx, playerModifier, 1, state):
                temp = self.__miniMaxRecursive(state, self.__get_number_of_empty_row_and_col(state), alpha, beta, True)

            else:
                state.player1_turn = not state.player1_turn
                temp = self.__miniMaxRecursive(state, self.__get_number_of_empty_row_and_col(state), alpha, beta, False)
                state.player1_turn = not state.player1_turn

            if (temp > maxScore):
                maxScore = temp
                best_action_type = line_type
                best_action_idx = idx
            
            alpha = max(alpha, temp)

            if (beta <= alpha):
                break
            self.update_board_and_check_additional_turn(line_type, idx, playerModifier, -1, state)

        best_action_x = 0
        best_action_y = 0

        if (best_action_type == "row"):
            best_action_y = best_action_idx // 3
            best_action_x = best_action_idx - (best_action_y * 3)

        elif (best_action_type == "col"):
            best_action_y = best_action_idx // 4
            best_action_x = best_action_idx - (best_action_y * 4)

        if maxScore <= alpha:
            if not exist:
                self.__transposition_table[hash_state_result] = [-math.inf, math.inf]
            self.__transposition_table[hash_state_result][1] = maxScore
        
        if maxScore > alpha and maxScore < beta:
            if not exist:
                self.__transposition_table[hash_state_result] = [-math.inf, math.inf]
            self.__transposition_table[hash_state_result][0] = maxScore
            self.__transposition_table[hash_state_result][1] = maxScore
        
        if maxScore >= beta:
            if not exist:
                self.__transposition_table[hash_state_result] = [-math.inf, math.inf]
            self.__transposition_table[hash_state_result][0] = maxScore

        hash_state = self.__hash_state(state)
        
        return GameAction(best_action_type, (best_action_x, best_action_y))
        
    def __miniMaxRecursive(self, state : GameStateCopy, depth : int, alpha : float, beta : float, maximizing : bool):
        self.number_of_nodes += 1

        playerModifier = -1 if state.player1_turn else 1
        number_of_nodes_limit = 20000

        hash_state_result = self.__hash_state(state)
        exist, transposition_table_result = self.__check_exist_in_transposition_table(state)
        if exist:
            if transposition_table_result[0] >= beta:
                self.number_of_cuts += 1
                return transposition_table_result[0]

            elif transposition_table_result[1] <= alpha:
                self.number_of_cuts += 1
                return transposition_table_result[1]

            alpha = max(alpha, transposition_table_result[0])
            beta = min(beta, transposition_table_result[1])

        if (depth <= self.__limit_depth or self.number_of_nodes >= number_of_nodes_limit):
            return self.__utility_function(state, maximizing)
        
        row_or_columns, indexes = self.__get_best_moves(state)
        
        if (maximizing):
            score = -10
            
            for i, idx in enumerate(indexes):
                line_type = "row" if row_or_columns[i] == 0 else "col"
                temp = 0
                if self.update_board_and_check_additional_turn(line_type, idx, playerModifier, 1, state):
                    temp = self.__miniMaxRecursive(state, depth - 1, alpha, beta, True)
                    score = max(temp, score)
                    alpha = max(alpha, temp)
                else:
                    state.player1_turn = not state.player1_turn
                    temp = self.__miniMaxRecursive(state, depth - 1, alpha, beta, False)
                    state.player1_turn = not state.player1_turn

                    score = max(temp, score)
                    alpha = max(alpha, temp)
                
                self.update_board_and_check_additional_turn(line_type, idx, playerModifier, -1, state)

                if (beta <= alpha):
                    break
                
                if (self.number_of_nodes >= number_of_nodes_limit):
                    break

            return score

        else:
            score = 10
            for i, idx in enumerate(indexes):
                line_type = "row" if row_or_columns[i] == 0 else "col"
                temp = 0

                if self.update_board_and_check_additional_turn(line_type, idx, playerModifier, 1, state):
                    temp = self.__miniMaxRecursive(state, depth - 1, alpha, beta, False)
                    score = min(temp, score)
                    beta = min(beta, temp)
                else:
                    state.player1_turn = not state.player1_turn
                    temp = self.__miniMaxRecursive(state, depth - 1, alpha, beta, True)
                    state.player1_turn = not state.player1_turn

                    score = min(temp, score)
                    beta = min(beta, temp)
                
                self.update_board_and_check_additional_turn(line_type, idx, playerModifier, -1, state)

                if (beta <= alpha):
                    break

                if (self.number_of_nodes >= number_of_nodes_limit):
                    break

            if score <= alpha:
                if not exist:
                    self.__transposition_table[hash_state_result] = [-math.inf, math.inf]
                self.__transposition_table[hash_state_result][1] = score
            
            if score > alpha and score < beta:
                if not exist:
                    self.__transposition_table[hash_state_result] = [-math.inf, math.inf]
                self.__transposition_table[hash_state_result][0] = score
                self.__transposition_table[hash_state_result][1] = score
            
            if score >= beta:
                if not exist:
                    self.__transposition_table[hash_state_result] = [-math.inf, math.inf]
                self.__transposition_table[hash_state_result][0] = score

            return score    

    def __utility_function(self, state: GameStateCopy, maximizing : bool):
        score = 0
        for elements in state.board_status:
            for element in elements:
                if abs(element) == 4:
                    if (element < 0 and state.player1_turn) or (element > 0 and not state.player1_turn):
                        score += 1
                    elif (element > 0 and state.player1_turn) or (element < 0 and not state.player1_turn):
                        score -= 1
        
        minimax_modifier = 1 if maximizing else -1
        return score * minimax_modifier
    
    def __hash_state(self, state: GameStateCopy):
        player2_turn = False
        if not state.player1_turn:
            player2_turn = True
            state.player1_turn = True
            state.board_status *= -1

        hash_result_original = self.__hash_function(state)
        hash_result = hash_result_original

        if hash_result not in self.__transposition_table:
            state.board_status = np.flipud(state.board_status)
            state.row_status = np.flipud(state.row_status)
            state.col_status = np.flipud(state.col_status)

            hash_result = self.__hash_function(state)

            state.board_status = np.flipud(state.board_status)
            state.row_status = np.flipud(state.row_status)
            state.col_status = np.flipud(state.col_status)
        
        if hash_result not in self.__transposition_table:
            state.board_status = np.fliplr(state.board_status)
            state.row_status = np.fliplr(state.row_status)
            state.col_status = np.fliplr(state.col_status)

            hash_result = self.__hash_function(state)
            
            state.board_status = np.fliplr(state.board_status)
            state.row_status = np.fliplr(state.row_status)
            state.col_status = np.fliplr(state.col_status)
        
        # Diagonal menurun
        if hash_result not in self.__transposition_table:
            col_status = np.fliplr(np.rot90(state.row_status, axes=(1, 0)))
            row_status = np.fliplr(np.rot90(state.col_status, axes=(1, 0)))
            state.board_status = np.fliplr(np.rot90(state.board_status, axes=(1, 0)))
            state.row_status = row_status
            state.col_status = col_status

            hash_result = self.__hash_function(state)
            
            col_status = np.rot90(np.fliplr(state.row_status))
            row_status = np.rot90(np.fliplr(state.col_status))
            state.board_status = np.rot90(np.fliplr(state.board_status))
            state.row_status = row_status
            state.col_status = col_status
        
        # Diagonal menaik
        if hash_result not in self.__transposition_table:
            col_status = np.fliplr(np.rot90(state.row_status))
            row_status = np.fliplr(np.rot90(state.col_status))
            state.board_status = np.fliplr(np.rot90(state.board_status))
            state.row_status = row_status
            state.col_status = col_status

            hash_result = self.__hash_function(state)
            
            col_status = np.rot90(np.fliplr(state.row_status), axes=(1, 0))
            row_status = np.rot90(np.fliplr(state.col_status), axes=(1, 0))
            state.board_status = np.rot90(np.fliplr(state.board_status), axes=(1, 0))
            state.row_status = row_status
            state.col_status = col_status
        
        if player2_turn:
            state.player1_turn = False
            state.board_status *= -1
        
        if hash_result not in self.__transposition_table:
            return hash_result_original
        else:
            return hash_result
    
    def __hash_function(self, state: GameStateCopy):
        row_status_hash = 0
        col_status_hash = 0
        board_status_hash = 0

        multiplier = 1
        for elements in state.row_status:
            for element in elements:
                if element == 1:
                    row_status_hash += (multiplier * 1)
                    
                multiplier *= 10
        
        multiplier = 1
        for elements in state.col_status:
            for element in elements:
                if element == 1:
                    col_status_hash += (multiplier * 1)

                multiplier *= 10

        multiplier = 1
        for elements in state.board_status:
            for element in elements:
                if element < 0:
                    board_status_hash += (multiplier * -1 * int(element) * 1)
                elif element > 0:
                    board_status_hash += (multiplier * int(element) * 2)

                multiplier *= 10
        
        return TranspositionTableKey(row_status_hash, col_status_hash, board_status_hash)

    def __get_best_moves(self, state: GameStateCopy):
        row_or_columns = []
        indexes = []

        chain_moves_row_or_col, chain_moves_index, allow_additional_moves = self.__get_chain_best_moves(state)
        row_or_columns += chain_moves_row_or_col
        indexes += chain_moves_index

        if allow_additional_moves:
            additional_moves_row_or_col, additional_moves_index = self.__get_spiral_moves(state, row_or_columns, indexes)
            row_or_columns += additional_moves_row_or_col
            indexes += additional_moves_index
            pass

        return row_or_columns, indexes

    
    def __get_spiral_moves(self, state: GameStateCopy, exist_row_or_columns: list, exist_indexes: list):

        row_or_columns_first = []
        indexes_first = []
        row_or_columns_last = []
        indexes_last = []

        row_status = state.row_status
        col_status = state.col_status
        board_status = state.board_status

        for idx, row_or_column in enumerate(self.__spiral_move_row_or_column_helper):
            position = self.__spiral_move_row_or_column_index[idx]

            if row_or_column == 0:  # row            
                i = position // 3
                j = position - (i * 3)
                if row_status[i][j] == 0:    
                    try_idx = -1
                    loop_count = 0
                    exist = False
                    while loop_count < 2:
                        try:
                            try_idx = exist_indexes.index(position, try_idx + 1)
                            if exist_row_or_columns[try_idx] == row_or_column:
                                exist = True
                                break
                        except:
                            pass
                        loop_count += 1
                    
                    if exist:
                        break

                    first = True
                    if i != 0:
                        if abs(board_status[i - 1][j]) == 2:
                            first = False
                    
                    if first and i != 3:
                        if abs(board_status[i][j]) == 2:
                            first = False
                    
                    if first:
                        row_or_columns_first.append(row_or_column)
                        indexes_first.append(position)
                    else:
                        row_or_columns_last.append(row_or_column)
                        indexes_last.append(position)

            elif row_or_column == 1:  # col
                i = position // 4
                j = position - (i * 4)

                if col_status[i][j] == 0:
                    try_idx = -1
                    loop_count = 0
                    exist = False
                    while loop_count < 2:
                        try:
                            try_idx = exist_indexes.index(position, try_idx + 1)
                            if exist_row_or_columns[try_idx] == row_or_column:
                                exist = True
                                break
                        except:
                            pass
                        loop_count += 1
                    
                    if exist:
                        break

                    first = True
                    if j != 0:
                        if abs(board_status[i][j - 1]) == 2:
                            first = False
                    
                    if first and j != 3:
                        if abs(board_status[i][j]) == 2:
                            first = False
                    
                    if first:
                        row_or_columns_first.append(row_or_column)
                        indexes_first.append(position)
                    else:
                        row_or_columns_last.append(row_or_column)
                        indexes_last.append(position)
        
        row_or_columns = row_or_columns_first + row_or_columns_last
        indexes = indexes_first + indexes_last
 
        self.__cut_corners(row_or_columns, indexes)

        return row_or_columns, indexes
    
    def __cut_corners(self, row_or_columns: list, indexes: list):
        # Pick the topmost corner only if both are exist

        # Check top left corner
        index_top = -1
        index_bottom = -1
        loop_count = 0
        exist = False
        while loop_count < 2:
            try:
                index_top = indexes.index(0, index_top + 1)
                if row_or_columns[index_top] == 0:
                    exist = True
                    break
            except:
                pass
            loop_count += 1

        if exist:
            loop_count = 0
            exist = False
            while loop_count < 2:
                try:
                    index_bottom = indexes.index(0, index_bottom + 1)
                    if row_or_columns[index_bottom] == 1:
                        del row_or_columns[index_bottom]
                        del indexes[index_bottom]
                        break
                except:
                    pass
                loop_count += 1
        
        # Check top right corner
        index_top = -1
        index_bottom = -1
        loop_count = 0
        exist = False
        while loop_count < 2:
            try:
                index_top = indexes.index(2, index_top + 1)
                if row_or_columns[index_top] == 0:
                    exist = True
                    break
            except:
                pass
            loop_count += 1

        if exist:
            loop_count = 0
            exist = False
            while loop_count < 2:
                try:
                    index_bottom = indexes.index(3, index_bottom + 1)
                    if row_or_columns[index_bottom] == 1:
                        del row_or_columns[index_bottom]
                        del indexes[index_bottom]
                        break
                except:
                    pass
                loop_count += 1
        
        # Check bottom right corner
        index_top = -1
        index_bottom = -1
        loop_count = 0
        exist = False
        while loop_count < 2:
            try:
                index_top = indexes.index(11, index_top + 1)
                if row_or_columns[index_top] == 1:
                    exist = True
                    break
            except:
                pass
            loop_count += 1

        if exist:
            loop_count = 0
            exist = False
            while loop_count < 2:
                try:
                    index_bottom = indexes.index(11, index_bottom + 1)
                    if row_or_columns[index_bottom] == 0:
                        del row_or_columns[index_bottom]
                        del indexes[index_bottom]
                        break
                except:
                    pass
                loop_count += 1
        
        # Check bottom left corner
        index_top = -1
        index_bottom = -1
        loop_count = 0
        exist = False
        while loop_count < 2:
            try:
                index_top = indexes.index(8, index_top + 1)
                if row_or_columns[index_top] == 1:
                    exist = True
                    break
            except:
                pass
            loop_count += 1

        if exist:
            loop_count = 0
            exist = False
            while loop_count < 2:
                try:
                    index_bottom = indexes.index(9, index_bottom + 1)
                    if row_or_columns[index_bottom] == 0:
                        del row_or_columns[index_bottom]
                        del indexes[index_bottom]
                        break
                except:
                    pass
                loop_count += 1

    def __get_chain_best_moves(self, state: GameStateCopy):
        number_of_chains = 0
        moves_row_or_col = []
        moves_index = []
        allow_additional_moves = True

        board_status = state.board_status

        part_of_chains = set()
        for i in range(len(board_status)):
            for j in range(len(board_status[i])):
                if ((i, j) not in part_of_chains):
                    if abs(board_status[i][j]) == 3:
                        allow_additional_moves = allow_additional_moves and self.__check_chain((i, j), state, moves_row_or_col, moves_index, 0, part_of_chains)
                        number_of_chains += 1
        
        if number_of_chains > 1:
            allow_additional_moves = False
        return moves_row_or_col, moves_index, allow_additional_moves

    def __check_chain(self, idx_board: tuple, state: GameStateCopy, row_or_columns: list, indexes: list, length: int, part_of_chains: set):
        length += 1
        row_status = state.row_status
        col_status = state.col_status
        board_status = state.board_status
        
        part_of_chains.add(idx_board)

        # 0: up, 1: right, 2: down, 3: left
        direction_type = 0
        found = False
        idx_line, line_type = self.__get_up_idx_and_type(idx_board)

        if row_status[idx_line[0]][idx_line[1]] == 0:
            found = True
        
        if not found:
            idx_line, line_type = self.__get_right_idx_and_type(idx_board)
            if col_status[idx_line[0]][idx_line[1]] == 0:
                found = True
                direction_type = 1

        if not found:
            idx_line, line_type = self.__get_down_idx_and_type(idx_board)
            if row_status[idx_line[0]][idx_line[1]] == 0:
                found = True
                direction_type = 2
        
        if not found:
            idx_line, line_type = self.__get_left_idx_and_type(idx_board)
            if col_status[idx_line[0]][idx_line[1]] == 0:
                found = True
                direction_type = 3
        
        idx_next_box = (idx_line[0] - 1, idx_line[1])
        
        if direction_type == 1:
            idx_next_box = (idx_line[0], idx_line[1] + 1)
        
        elif direction_type == 2:
            idx_next_box = (idx_line[0] + 1, idx_line[1])

        elif direction_type == 3:
            idx_next_box = (idx_line[0], idx_line[1] - 1)
        

        if length < 2:
            row_or_columns.append(line_type)
            if line_type == 0:
                idx_next_move = idx_line[0] * 3 + idx_line[1]
                indexes.append(idx_next_move)
            elif line_type == 1:
                idx_next_move = idx_line[0] * 4 + idx_line[1]
                indexes.append(idx_next_move)
        
        if idx_next_box[0] < 0 or idx_next_box[0] > 2 or idx_next_box[1] < 0 or idx_next_box[1] > 2 or abs(board_status[idx_next_box[0]][idx_next_box[1]]) != 2 or abs(board_status[idx_next_box[0]][idx_next_box[1]]) != 3:
            if length == 1:
                return True
            elif length == 2:
                if abs(board_status[idx_board[0]][idx_board[1]]) != 3:
                    row_or_columns.append(line_type)
                    if line_type == 0:
                        idx_next_move = idx_line[0] * 3 + idx_line[1]
                        indexes.append(idx_next_move)
                    elif line_type == 1:
                        idx_next_move = idx_line[0] * 4 + idx_line[1]
                        indexes.append(idx_next_move)
                return True
            else:
                return False
        else:
            if length == 2:
                return False
            else:
                return self.__check_chain(idx_next_box, state, row_or_columns, indexes, length)
        
    def __check_exist_in_transposition_table(self, state: GameStateCopy):
        exist = False
        transposition_table_result = None

        hash_state_result = self.__hash_state(state)

        if hash_state_result in self.__transposition_table:
            transposition_table_result = self.__transposition_table[hash_state_result]
            exist = True

        return exist, transposition_table_result

    def __get_up_idx_and_type(self, idx_board: tuple):
        return (idx_board[0], idx_board[1]), 0
    
    def __get_down_idx_and_type(self, idx_board: tuple):
        return (idx_board[0] + 1, idx_board[1]), 0
    
    def __get_right_idx_and_type(self, idx_board: tuple):
        return (idx_board[0], idx_board[1] + 1), 1
    
    def __get_left_idx_and_type(self, idx_board: tuple):
        return (idx_board[0], idx_board[1]), 1
