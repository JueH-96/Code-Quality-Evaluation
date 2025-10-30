def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read the 3x3 grid in row-major order.
    grid = tuple(int(x) for x in data)
    
    # We represent the board as a tuple of 9 integers.
    # 0 = white (empty), 1 = red (Takahashi), 2 = blue (Aoki).
    start_board = (0,) * 9

    # Predefine all possible winning (three-in-a-row) lines: 3 rows, 3 columns, 2 diagonals.
    win_lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8), # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8), # columns
        (0, 4, 8), (2, 4, 6)             # diagonals
    ]
    
    def check_win(board):
        """
        Check if the current board has a triple.
        Returns:
          1 if Takahashi (red) forms a triple,
          2 if Aoki (blue) forms a triple,
          0 if no triple exists.
        """
        for (i, j, k) in win_lines:
            if board[i] != 0 and board[i] == board[j] == board[k]:
                return board[i]
        return 0

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def minimax(board, turn, tscore, ascore):
        """
        Recursive function using minimax with memoization.
        board: tuple of 9 integers representing the current board.
        turn: 0 for Takahashi, 1 for Aoki.
        tscore, ascore: current scores for Takahashi and Aoki respectively.
        
        Returns True if the position (with perfect play from here) eventually
        results in a win for Takahashi; otherwise False (meaning Aoki wins).
        """
        # First check for an immediate winning triple.
        winner = check_win(board)
        if winner == 1:
            return True   # Takahashi wins immediately.
        if winner == 2:
            return False  # Aoki wins immediately.
        
        # If the board is full and no triple was created, decide by score.
        if all(cell != 0 for cell in board):
            return tscore > ascore  # Because sum is odd, tie is impossible.

        # Enumerate possible moves:
        if turn == 0:
            # Takahashi's turn: he chooses a white cell, gets grid[i] points,
            # colors it red (1), and then it becomes Aoki's turn.
            for i in range(9):
                if board[i] == 0:
                    new_tscore = tscore + grid[i]
                    new_board = list(board)
                    new_board[i] = 1  # Mark this cell red.
                    new_board = tuple(new_board)
                    # If there is any move that leads to a winning outcome, Takahashi will choose it.
                    if minimax(new_board, 1, new_tscore, ascore):
                        return True
            return False  # If no winning move is available, Takahashi loses.
        else:
            # Aoki's turn: he chooses a white cell, he gets grid[i] points,
            # colors it blue (2), and then it becomes Takahashi's turn.
            for i in range(9):
                if board[i] == 0:
                    new_ascore = ascore + grid[i]
                    new_board = list(board)
                    new_board[i] = 2  # Mark this cell blue.
                    new_board = tuple(new_board)
                    # Aoki will choose a move that forces a win for him (i.e. Takahashi loses).
                    if not minimax(new_board, 0, tscore, new_ascore):
                        return False
            return True  # If Aoki cannot force a win, then Takahashi wins.

    # Start from the initial state: an empty board, Takahashi's turn with both scores 0.
    result = minimax(start_board, 0, 0, 0)
    sys.stdout.write("Takahashi" if result else "Aoki")

if __name__ == '__main__':
    main()