from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import random
import numpy as np

class JasonBot(Bot):
    def get_action(self, state: GameState) -> GameAction:
        pass

    def get_random_action(self, state: GameState) -> GameAction:
        if random.random() < 0.5:
            return self.get_random_row_action(state)
        else:
            return self.get_random_col_action(state)

    def get_random_row_action(self, state: GameState) -> GameAction:
        position = self.get_random_position_with_zero_value(state.row_status)
        return GameAction("row", position)

    def get_random_position_with_zero_value(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape

        x = -1
        y = -1
        valid = False
        
        while not valid:
            x = random.randrange(0, nx)
            y = random.randrange(0, ny)
            valid = matrix[y, x] == 0
        
        return (x, y)

    def get_random_col_action(self, state: GameState) -> GameAction:
        position = self.get_random_position_with_zero_value(state.col_status)
        return GameAction("col", position)
