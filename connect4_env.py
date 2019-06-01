import numpy as np
import os


class Connect4env:

    def __init__(self, width=7, height=6):
        self.width = width
        self.height = height
        self.board = np.full((self.width, self.height), 0)

    def get_current_player(self, state=None):
        state = state if state is not None else self.board.copy()

        # Unique will contain the values that are currently on
        # the board (most likely [0, 1, 2]), and counts will
        # contain how many times each vaule occurs on the board
        unique, counts = np.unique(state, return_counts=True)

        player1_count = counts[np.where(unique == 1)]
        player2_count = counts[np.where(unique == 2)]

        if len(player1_count) == 0:
            player1_count = 0
        if len(player2_count) == 0:
            player2_count = 0
        if player1_count > player2_count:
            return 2
        else:
            return 1

    def step(self, col_idx):
        step_player = self.get_current_player()

        # In case an invalid column index is provided
        if col_idx >= self.width:
            state = self.board.copy()
            reward = -1
            result = -1
            return state, reward, result

        # In case the given column is full
        row_idx = np.argmin(self.board, 1)[col_idx]
        if self.board[col_idx, row_idx] != 0:
            state = self.board.copy()
            reward = -1
            result = -1
            return state, reward, result

        self.board[col_idx, row_idx] = step_player
        state = self.board.copy()
        result = self._judge(col_idx, row_idx)
        if result == step_player:
            reward = 1
        else:
            reward = 0

        return state, reward, result

    # Determine if the player that just played is the winner
    def _judge(self, col_idx, row_idx):
        result = 0
        player = self.board[col_idx, row_idx]

        # Vertical: Just check the 3 pieces below the most recent
        if row_idx >= 3:
            check_range = self.board[col_idx, np.arange(row_idx - 3, row_idx + 1)]
            if len(check_range[check_range != player]) == 0:
                return player

        # Horizontal
        idx = col_idx
        # Find the left most piece that is the same as the current player
        while 0 <= idx < self.width and self.board[idx, row_idx] == player:
            idx -= 1
        idx += 1
        # From that piece, check the 3 pieces to its right
        if idx + 4 <= self.width:
            check_range = self.board[np.arange(idx, idx + 4), row_idx]
            if len(check_range[check_range != player]) == 0:
                return player

        # Main Diagonal
        c_idx = col_idx
        r_idx = row_idx
        # Go up the main diagonal to find the current player's highest piece
        while 0 <= c_idx < self.width and 0 <= r_idx < self.height and self.board[c_idx, r_idx] == player:
            c_idx -= 1
            r_idx += 1
        c_idx += 1
        r_idx -= 1
        # From that piece, check the 3 pieces down the diagonal
        if r_idx >= 3 and c_idx + 4 <= self.width:
            check_range = np.diag(np.rot90(self.board[c_idx:c_idx + 4, r_idx - 3:r_idx + 1]))
            if len(check_range[check_range != player]) == 0:
                return player

        # Other Diagonal
        c_idx = col_idx
        r_idx = row_idx
        # Go down the second diagonal to find the current player's lowest piece
        while 0 <= c_idx < self.width and 0 <= r_idx < self.height and self.board[c_idx, r_idx] == player:
            c_idx -= 1
            r_idx -= 1
        c_idx += 1
        r_idx += 1
        # From that piece, check the 3 pieces up the diagonal
        if c_idx + 4 <= self.width and r_idx + 4 <= self.height:
            check_range = np.diag(self.board[c_idx:c_idx + 4, r_idx:r_idx + 4])
            if len(check_range[check_range != player]) == 0:
                return player

        # Check for draw
        if np.sum(self.board != 0) == self.width * self.height:
            return 3

        return result

    def simulate(self, test_state, col_idx):
        # Backup current state and player
        snapshot = self.board.copy()
        # play on given state
        self.board = test_state.copy()
        state, reward, result = self.step(col_idx)
        # Restore state and current player
        self.board = snapshot

        return state, reward, result

    def get_all_next_actions(self):
        return [action for action in range(self.width)]

    # Retursns the actions in this fashion:
    # [1, 1, 1, 0, 0, 1, 1], meaning that actions 0, 1, 2, 5, 6
    # are valid but actions 3 and 4 are not
    def get_valid_actions(self, state=None):
        state = state if state is not None else self.board.copy()
        actions = []
        for col_idx in range(self.width):
            if np.min(state, 1)[col_idx] > 0:
                actions.append(0)
            else:
                actions.append(1)
        return actions

    def reset(self):
        self.board = np.full((self.width, self.height), 0)

    def to_str(self, board=None):
        string = os.linesep
        board = board if board is not None else self.board
        b = np.rot90(board).reshape(self.width * self.height)
        for idx, c in enumerate(b):
            c = int(c)
            if (idx + 1) % self.width > 0:
                string = '{}{} '.format(string, c)
            else:
                string = '{}{}{}'.format(string, c, os.linesep)
        return string

    def print(self, board=None):
        print(self.to_str(board))

    def get_state(self):
        return self.board.copy().astype(dtype=np.float32)

    # Return the symmetric of the current state around the middle column
    def get_mirror_state(self, board=None):
        board = board if board is not None else self.board
        mirror = np.array(board)
        for col_idx in range(self.width):
            for row_idx in range(self.height):
                mirror[self.width - col_idx - 1, row_idx] = board[col_idx, row_idx]
        return mirror

    # Returns the inverse of the current state (1s are 2s and 2s are 1s)
    def get_inv_state(self, board=None):
        board = board if board is not None else self.board
        inv = board.copy()
        for col_idx in range(self.width):
            for row_idx in range(self.height):
                if inv[col_idx, row_idx] == 1:
                    inv[col_idx, row_idx] = 2
                elif inv[col_idx, row_idx] == 2:
                    inv[col_idx, row_idx] = 1
        return inv


def main():
    b = Connect4env()

    while True:
        player = b.get_current_player()
        b.print()
        col_idx = int(input('Player {}\'s turn. Please input the col number (1 to {}) you want to place your chip: '.format(player, b.width)))
        state, reward, result = b.step(col_idx - 1)
        if result < 0:
            print('Your input is invalid.')
        elif result == 0:
            pass
        elif result == 3:
            print('Draw game!!!')
            break
        else:
            print('Player', player, 'won!!!')
            b.print()
            break


if __name__ == '__main__':
    main()
