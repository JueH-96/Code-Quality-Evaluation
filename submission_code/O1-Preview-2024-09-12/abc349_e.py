# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1000000)

    A = []
    for _ in range(3):
        A.extend(map(int, sys.stdin.readline().split()))
    total_sum = sum(A)
    assert total_sum % 2 == 1  # sum is odd

    N = 9  # number of cells

    # Precompute lines (winning conditions)
    lines = [
        (0,1,2),  # rows
        (3,4,5),
        (6,7,8),
        (0,3,6),  # columns
        (1,4,7),
        (2,5,8),
        (0,4,8),  # diagonals
        (2,4,6),
    ]

    from functools import lru_cache

    def check_win(board_state, player):
        # Check if player (1 for Takahashi, 2 for Aoki) has a line
        for line in lines:
            if all(board_state[i] == player for i in line):
                return True
        return False

    @lru_cache(None)
    def minimax(board_state, turn, score_diff):
        # Check for winning condition
        if turn == 'A':
            # Last move was made by Takahashi
            if check_win(board_state, 1):  # Takahashi wins
                return float('inf')
        else:
            # Last move was made by Aoki
            if check_win(board_state, 2):  # Aoki wins
                return float('-inf')
        # Check if game over
        if all(cell != 0 for cell in board_state):
            # Game over, compare the scores
            if score_diff > 0:
                return float('inf')  # Takahashi wins
            elif score_diff < 0:
                return float('-inf')  # Aoki wins
            else:
                # Since total sum is odd, this should not happen
                # But in case it does, decide according to problem statement
                return float('-inf')  # Aoki wins because Takahashi went first?
        # Generate possible moves
        if turn == 'T':
            best = float('-inf')
            for i in range(N):
                if board_state[i] == 0:
                    new_board = list(board_state)
                    new_board[i] = 1  # Takahashi paints red
                    new_score_diff = score_diff + A[i]
                    val = minimax(tuple(new_board), 'A', new_score_diff)
                    if val == float('inf'):
                        return float('inf')  # Takahashi can force a win
                    if val > best:
                        best = val
            return best
        else:
            best = float('inf')
            for i in range(N):
                if board_state[i] == 0:
                    new_board = list(board_state)
                    new_board[i] = 2  # Aoki paints blue
                    new_score_diff = score_diff - A[i]
                    val = minimax(tuple(new_board), 'T', new_score_diff)
                    if val == float('-inf'):
                        return float('-inf')  # Aoki can force a win
                    if val < best:
                        best = val
            return best

    initial_board = tuple([0]*9)  # All cells are unpainted
    result = minimax(initial_board, 'T', 0)
    if result == float('inf'):
        print("Takahashi")
    else:
        print("Aoki")

threading.Thread(target=main).start()