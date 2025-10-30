def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read the 3x3 grid values.
    # points[i] corresponds to cell i in row-major order.
    points = [int(x) for x in data]
    if len(points) != 9:
        return

    # The board will be represented as a tuple of 9 integers:
    # 0 means white (uncolored), 1 means red (Takahashi), 2 means blue (Aoki).
    initial_board = (0,) * 9

    # Pre-calculate the list of winning lines (rows, columns, diagonals).
    lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    # Function to check if the board has a three-in-a-row formation.
    # Returns 1 if red (Takahashi) has it, 2 if blue (Aoki) has it, and None otherwise.
    def check_win(board):
        for a, b, c in lines:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        return None

    from functools import lru_cache

    # Our dfs function:
    # Parameters:
    #   board: tuple representing the board state.
    #   turn: 0 for Takahashi's turn (red), 1 for Aoki's turn (blue).
    #   diff: current score difference = (Takahashi's total) - (Aoki's total).
    # Returns: 1 if with optimal play starting from this state Takahashi wins, otherwise 0.
    @lru_cache(maxsize=None)
    def dfs(board, turn, diff):
        # First, if any winning formation is already on the board, then the game is over.
        win = check_win(board)
        if win is not None:
            # If formation of red, Takahashi wins; if formation of blue, Aoki wins.
            return 1 if win == 1 else 0

        # Count white cells.
        if board.count(0) == 0:
            # Game over by turns exhausted; winner determined by score difference.
            return 1 if diff > 0 else 0

        # Identify whose turn it is.
        # turn = 0: Takahashi (maximizer, wants outcome 1)
        # turn = 1: Aoki (minimizer, wants outcome 0)
        if turn == 0:
            # Takahashi's turn. He will try to select a move that eventually leads to a win (value 1).
            best = 0  # Worst-case: lose (0)
            for i in range(9):
                if board[i] == 0:
                    # Create a new board with cell i colored red (1).
                    board_list = list(board)
                    board_list[i] = 1
                    new_board = tuple(board_list)
                    new_diff = diff + points[i]
                    # Check immediately in the next state if this move gives formation.
                    child_outcome = dfs(new_board, 1, new_diff)
                    if child_outcome == 1:
                        # Found a winning move.
                        return 1
                    # Otherwise, best remains what it is.
            return best
        else:
            # Aoki's turn. He wants to force a win for himself, i.e. outcome 0.
            worst = 1  # Worst-case from Takahashi's perspective.
            for i in range(9):
                if board[i] == 0:
                    board_list = list(board)
                    board_list[i] = 2  # Blue for Aoki.
                    new_board = tuple(board_list)
                    new_diff = diff - points[i]
                    child_outcome = dfs(new_board, 0, new_diff)
                    if child_outcome == 0:
                        # Aoki found a move that ensures his win.
                        return 0
                    # Otherwise, worst remains 1.
            return worst

    result = dfs(initial_board, 0, 0)
    if result == 1:
        sys.stdout.write("Takahashi")
    else:
        sys.stdout.write("Aoki")
    
main()