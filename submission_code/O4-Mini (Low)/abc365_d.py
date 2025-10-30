def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()

    # Map moves to integers: 0 = R, 1 = P, 2 = S
    char_to_int = {'R': 0, 'P': 1, 'S': 2}

    # dp_prev[m] = max wins up to previous game, ending with our move m
    # We'll use a large negative for impossible states
    NEG_INF = -10**9
    dp_prev = [NEG_INF] * 3

    # Initialize for the first game
    a0 = char_to_int[S[0]]
    # We can either draw (play a0) or win (play (a0+1)%3)
    draw_move = a0
    win_move = (a0 + 1) % 3

    dp_prev[draw_move] = 0
    dp_prev[win_move] = 1

    # Process games 2..N
    for i in range(1, N):
        ai = char_to_int[S[i]]
        draw_m = ai
        win_m = (ai + 1) % 3

        dp_curr = [NEG_INF] * 3
        # For each allowed move this round, consider transitions
        for m in (draw_m, win_m):
            win_point = 1 if m == win_m else 0
            # We must switch from a different previous move
            best_prev = NEG_INF
            for prev_m in range(3):
                if prev_m != m:
                    best_prev = max(best_prev, dp_prev[prev_m])
            dp_curr[m] = best_prev + win_point

        dp_prev = dp_curr

    # The answer is the max over ending moves
    answer = max(dp_prev)
    print(answer)

if __name__ == "__main__":
    main()