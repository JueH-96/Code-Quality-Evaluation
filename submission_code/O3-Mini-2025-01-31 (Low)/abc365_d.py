def main():
    import sys
    # read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    s = data[1]

    # For each Aoki move, allowed moves for Takahashi (must not lose):
    # If Aoki plays:
    #  'R': then allowed moves for Takahashi are:
    #         'R' (draw, wins = 0)
    #         'P' (win, wins = 1)
    #  'P': allowed moves:
    #         'P' (draw, wins = 0)
    #         'S' (win, wins = 1)
    #  'S': allowed moves:
    #         'S' (draw, wins = 0)
    #         'R' (win, wins = 1)
    allowed_moves = {
        'R': [('R', 0), ('P', 1)],
        'P': [('P', 0), ('S', 1)],
        'S': [('S', 0), ('R', 1)]
    }

    # We need to maximize the total win count subject to:
    #  1. No game Takahashi loses (so he must choose a move from allowed_moves for that round)
    #  2. Takahashi's move in the i-th game is different from his move in the (i+1)-th game.
    #
    # We'll use dynamic programming to track possible states.
    # Define dp[i][m] to be the maximum wins up to game i if Takahashi chooses move m in game i.
    # Moves can be R, P, S.
    
    # Initialization for game0
    dp = {'R': -10**9, 'P': -10**9, 'S': -10**9}
    first_aoki = s[0]
    for move, win in allowed_moves[first_aoki]:
        dp[move] = max(dp[move], win)
    
    # Process rounds 2 ... n (indexing 1 to n-1)
    for i in range(1, n):
        new_dp = {'R': -10**9, 'P': -10**9, 'S': -10**9}
        aoki_move = s[i]
        for move, win in allowed_moves[aoki_move]:
            # We want to choose the best dp value from previous round, 
            # but we cannot choose a move same as our current move (non-equality constraint).
            best_prev = -10**9
            for prev in ['R', 'P', 'S']:
                if prev != move:
                    best_prev = max(best_prev, dp[prev])
            if best_prev != -10**9:
                new_dp[move] = max(new_dp[move], best_prev + win)
        dp = new_dp

    answer = max(dp.values())
    sys.stdout.write(str(answer))

if __name__ == "__main__":
    main()