# YOUR CODE HERE

import sys
sys.setrecursionlimit(10**7)

def main():
    # Read input
    A = []
    for _ in range(3):
        A.extend(list(map(int, sys.stdin.readline().split())))
    # A is now a length-9 list: A[0],A[1],A[2] = row0, etc.

    # Precompute winning lines (indices in the 1D list of length 9):
    lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # columns
        (0,4,8), (2,4,6)            # diagonals
    ]

    # check_win(board):
    #   returns +1 if there's a line of three "1"s (T's),
    #           -1 if there's a line of three "2"s (A's),
    #            0 otherwise
    def check_win(board):
        for (x, y, z) in lines:
            if board[x] == board[y] == board[z] != 0:
                return +1 if board[x] == 1 else -1
        return 0

    # Minimax with alpha-beta:
    # board: list of 9 cells, each 0 (empty),1(T),2(A)
    # tscore, ascore: current scores for T and A
    # player: 1 => T's turn, 2 => A's turn
    # alpha, beta: alpha-beta bounds
    # returns the "score difference" from T's perspective:
    #   +inf if T forced a 3-in-a-row
    #   -inf if A forced a 3-in-a-row
    #   else if board filled => (tscore - ascore)
    def dfs(board, tscore, ascore, player, alpha, beta):
        # Check immediate win by 3 in a row
        w = check_win(board)
        if w != 0:
            return float('inf') if w == +1 else float('-inf')
        
        # Check if no empty cells => game ends by score
        if all(c != 0 for c in board):
            return tscore - ascore
        
        if player == 1:
            # T's turn, tries to maximize (tscore - ascore)
            val = float('-inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = 1
                    child_val = dfs(board, tscore + A[i], ascore, 2, alpha, beta)
                    board[i] = 0
                    if child_val > val:
                        val = child_val
                    alpha = max(alpha, val)
                    if alpha >= beta:
                        break
            return val
        else:
            # A's turn, tries to minimize (tscore - ascore)
            val = float('inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = 2
                    child_val = dfs(board, tscore, ascore + A[i], 1, alpha, beta)
                    board[i] = 0
                    if child_val < val:
                        val = child_val
                    beta = min(beta, val)
                    if beta <= alpha:
                        break
            return val

    # Initial empty board
    board = [0]*9
    # T starts, alpha=-inf, beta=+inf
    result = dfs(board, 0, 0, 1, float('-inf'), float('inf'))
    # If result > 0 => T wins, < 0 => A wins, 0 is not possible here because sum of A is odd
    # but just in case, we handle 0 => T
    if result >= 0:
        print("Takahashi")
    else:
        print("Aoki")

# Call main
if __name__ == "__main__":
    main()