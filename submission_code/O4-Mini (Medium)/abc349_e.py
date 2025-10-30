def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    A = list(map(int, data))
    # flatten A to length 9, row-major
    # board cells: 0=white, 1=red(T), 2=blue(A)
    lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6)            # diags
    ]
    # check if there is a triple; return 1 if T wins, -1 if A wins, 0 otherwise
    def check_triple(board):
        for x,y,z in lines:
            v = board[x]
            if v != 0 and board[y] == v and board[z] == v:
                return 1 if v == 1 else -1
        return 0

    # dfs returns 1 if Takahashi wins under optimal play from this state, else -1
    from functools import lru_cache

    # We won't include scores in cache; rely on brute DFS without memo.
    def dfs(board, turn, t_score, a_score):
        # Check terminal by triple
        ct = check_triple(board)
        if ct != 0:
            return ct
        # Check full
        if all(c != 0 for c in board):
            return 1 if t_score > a_score else -1

        if turn == 1:
            # Takahashi's turn: he wants any move leading to win
            best = -1
            for i in range(9):
                if board[i] == 0:
                    board[i] = 1
                    res = dfs(board, 2, t_score + A[i], a_score)
                    board[i] = 0
                    if res == 1:
                        return 1
            return -1
        else:
            # Aoki's turn: he wants any move leading to his win (res == -1)
            for i in range(9):
                if board[i] == 0:
                    board[i] = 2
                    res = dfs(board, 1, t_score, a_score + A[i])
                    board[i] = 0
                    if res == -1:
                        return -1
            return 1

    board = [0]*9
    winner = dfs(board, 1, 0, 0)
    print("Takahashi" if winner == 1 else "Aoki")

if __name__ == "__main__":
    main()