def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    S = input().strip()

    # For each Aoki move we list the two allowed moves for Takahashi.
    # One that draws (0 win) and one that wins (1 win).
    allowed_moves = {
        'R': [('R', 0), ('P', 1)],  # Against Rock: draw with R, win with P.
        'P': [('P', 0), ('S', 1)],  # Against Paper: draw with P, win with S.
        'S': [('S', 0), ('R', 1)]   # Against Scissors: draw with S, win with R.
    }
    
    # dp[i][m] = maximum wins up to game i if Takahashi played move m in game i.
    # We only store states for allowed moves.
    # For game 0:
    dp = {}  # keys will be one of 'R', 'P', 'S'
    for move, win_val in allowed_moves[S[0]]:
        dp[move] = win_val

    # Process games 1 to N-1:
    for i in range(1, N):
        new_dp = {}
        # For the i-th game, only two moves are allowed based on Aoki's move S[i].
        for move, win_val in allowed_moves[S[i]]:
            best_previous = -10**9
            # We want to pick any previous move that is not equal to the current move.
            for prev_move, score in dp.items():
                if prev_move != move:
                    best_previous = max(best_previous, score)
            if best_previous > -10**8:  # if we found a valid previous state
                if move in new_dp:
                    new_dp[move] = max(new_dp[move], best_previous + win_val)
                else:
                    new_dp[move] = best_previous + win_val
        dp = new_dp

    answer = max(dp.values())
    sys.stdout.write(str(answer) + "
")

if __name__ == '__main__':
    main()