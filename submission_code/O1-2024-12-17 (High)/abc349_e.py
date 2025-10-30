def main():
    import sys
    data = list(map(int, sys.stdin.read().split()))
    # We have exactly 9 integers for the 3x3 grid:
    A = data  # length 9, in row-major order: A[0..2] for row0, A[3..5] row1, A[6..8] row2

    # Precompute an ordering of cell indices to try first, to help alpha-beta pruning:
    # - Taka (turn=0) tries cells in descending order of A[i]
    # - Aoki (turn=1) tries cells in ascending order of A[i]
    idx_desc = sorted(range(9), key=lambda i: A[i], reverse=True)
    idx_asc = sorted(range(9), key=lambda i: A[i])

    # Possible winning lines (indices in the 1D board array):
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
        (0, 4, 8), (2, 4, 6)              # diagonals
    ]

    # board[i] = None (white), 0 (red), or 1 (blue)
    board = [None] * 9

    def check_line(bd):
        """
        Returns +1 if there's a 3-in-a-row of red (Takahashi),
        -1 if there's a 3-in-a-row of blue (Aoki),
         0 otherwise.
        """
        for x, y, z in lines:
            if bd[x] is not None and bd[x] == bd[y] == bd[z]:
                # 0 = red => Taka, 1 = blue => Aoki
                return 1 if bd[x] == 0 else -1
        return 0

    def minimax(diff, turn, alpha, beta):
        """
        diff: current (Taka_score - Aoki_score)
        turn: 0 for Taka, 1 for Aoki
        alpha, beta: for alpha-beta pruning
        Returns +1 if, with perfect play from this state, Taka wins; -1 if Aoki wins.
        """

        # Check if the previous move just formed a 3-in-a-row:
        c = check_line(board)
        if c != 0:
            return c  # +1 => Taka, -1 => Aoki

        # If no white cells remain, decide by final score difference:
        if all(cell is not None for cell in board):
            return 1 if diff > 0 else -1  # sum of A is odd => no tie

        if turn == 0:
            # Taka's turn (try to maximize outcome from Taka's perspective)
            best_value = -2  # smaller than the minimum possible (-1)
            for i in idx_desc:
                if board[i] is None:
                    board[i] = 0  # red
                    val = minimax(diff + A[i], 1, alpha, beta)
                    board[i] = None
                    if val > best_value:
                        best_value = val
                    alpha = max(alpha, best_value)
                    if alpha >= beta:  # alpha-beta cutoff
                        break
            return best_value
        else:
            # Aoki's turn (minimize outcome from Taka's perspective)
            best_value = 2  # larger than the maximum possible (+1)
            for i in idx_asc:
                if board[i] is None:
                    board[i] = 1  # blue
                    val = minimax(diff - A[i], 0, alpha, beta)
                    board[i] = None
                    if val < best_value:
                        best_value = val
                    beta = min(beta, best_value)
                    if alpha >= beta:  # alpha-beta cutoff
                        break
            return best_value

    result = minimax(0, 0, -2, 2)
    if result == 1:
        print("Takahashi")
    else:
        print("Aoki")