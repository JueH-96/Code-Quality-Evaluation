def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read the 3x3 grid values.
    A = list(map(int, data))
    # The board has 9 cells. We represent each cell’s color as:
    #   0: white (unclaimed)
    #   1: red (Takahashi)
    #   2: blue (Aoki)
    # Initially, all cells are white.
    board = [0] * 9

    # Precomputed winning lines (triples of indices in the board=list of length 9)
    # Rows, columns, and the two diagonals.
    win_lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),       # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),       # columns
        (0, 4, 8), (2, 4, 6)                   # diagonals
    ]

    def check_win(board):
        """
        Checks if board has a winning line of a color.
        Returns:
          1 if three red cells (Takahashi) form a consecutive line.
          2 if three blue cells (Aoki) form a consecutive line.
          0 if no winning line exists.
        """
        for (i, j, k) in win_lines:
            if board[i] != 0 and board[i] == board[j] and board[j] == board[k]:
                return board[i]
        return 0

    # We use minimax with memoization.
    # game state: (board_tuple, turn, score_diff)
    # turn: 1 if it is Takahashi's turn, 2 if Aoki's turn.
    # score_diff: (Takahashi's total points - Aoki's total points)
    #
    # Terminal conditions:
    # 1. If a winning (three consecutive same color) formation is found,
    #    then the last mover wins immediately. (Red (1) means Takahashi wins; blue (2) means Aoki wins.)
    # 2. If no white cells remain, the scores are compared.
    #    score_diff > 0 means Takahashi wins, otherwise Aoki wins.
    # In our minimax utility, we return 1 if eventually Takahashi wins and -1 if Aoki wins.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(board_tuple, turn, score_diff):
        # Convert board_tuple to list for update.
        board_state = list(board_tuple)

        # Terminal check: if there is a winning line on board.
        winner = check_win(board_state)
        if winner:
            if winner == 1:
                return 1   # Takahashi (red) wins.
            else:
                return -1  # Aoki (blue) wins.

        # Terminal check: if board is full then compare score totals.
        if all(cell != 0 for cell in board_state):
            return 1 if score_diff > 0 else -1

        # If game is not over, then try all moves.
        if turn == 1:
            # Takahashi's turn: he wants to maximize outcome (seek outcome 1).
            best_val = -2  # lower than any possible outcome
            for i in range(9):
                if board_state[i] == 0:
                    new_board = board_state[:]
                    new_board[i] = 1  # mark cell with red.
                    # Update score: Takahashi adds A[i] to his score.
                    new_score_diff = score_diff + A[i]
                    # Check immediate winning condition for Takahashi.
                    if check_win(new_board) == 1:
                        outcome = 1
                    else:
                        outcome = dfs(tuple(new_board), 2, new_score_diff)
                    best_val = max(best_val, outcome)
                    if best_val == 1:
                        # Best possible outcome so stop early.
                        return best_val
            return best_val
        else:
            # Aoki's turn: he wants to minimize outcome (seek outcome -1).
            best_val = 2  # higher than any possible outcome
            for i in range(9):
                if board_state[i] == 0:
                    new_board = board_state[:]
                    new_board[i] = 2  # mark cell with blue.
                    # Update score: Aoki's score is subtracted.
                    new_score_diff = score_diff - A[i]
                    # Check immediate winning condition for Aoki.
                    if check_win(new_board) == 2:
                        outcome = -1
                    else:
                        outcome = dfs(tuple(new_board), 1, new_score_diff)
                    best_val = min(best_val, outcome)
                    if best_val == -1:
                        # Best possible outcome for Aoki so stop early.
                        return best_val
            return best_val

    # Start DFS from the initial state:
    # - board is all white (represented as tuple(board))
    # - Turn 1 means Takahashi goes first.
    # - Initial score difference is 0.
    result = dfs(tuple(board), 1, 0)
    print("Takahashi" if result == 1 else "Aoki")

if __name__ == '__main__':
    main()