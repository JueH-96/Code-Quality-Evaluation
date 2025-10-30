def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    S = data[1].strip()
    
    # Map characters to numbers for easier handling: R->0, P->1, S->2.
    # Note: In rock-paper-scissors, we can observe that the winning move for Aoki’s move 'x'
    # is given by (x+1)%3 if we set the order as R, P, S.
    char_to_int = {'R': 0, 'P': 1, 'S': 2}
    
    # Let dp[i][m] be the maximum wins up to round i with Takahashi's last move being m.
    # We only need to keep track of the previous round's dp, i.e., we use a one-dimensional array dp.
    # We use a very low number for impossible states.
    NEG_INF = -10**9
    dp = [NEG_INF, NEG_INF, NEG_INF]
    
    # Process the first game (round index 0).
    # For each game, Takahashi does not lose. That means if Aoki plays 'x', Takahashi's move
    # must be either 'x' (tie) or a winning move which is (x+1)%3.
    # So for round 0, allowed moves:
    a = char_to_int[S[0]]   # Aoki's move in round 0 as an integer.
    # Allowed moves are: tie move (a) and winning move ((a+1)%3).
    allowed_moves = (a, (a + 1) % 3)
    new_dp = [NEG_INF, NEG_INF, NEG_INF]
    for m in allowed_moves:
        # If m is the winning move then we score 1 win; otherwise 0.
        cur_score = 1 if m == (a + 1) % 3 else 0
        new_dp[m] = cur_score
    dp = new_dp
    
    # Process rounds 1 to n-1.
    for i in range(1, n):
        a = char_to_int[S[i]]
        allowed_moves = (a, (a + 1) % 3)  # allowed moves: tie or win.
        new_dp = [NEG_INF, NEG_INF, NEG_INF]
        for m in allowed_moves:
            # Calculate the win score for current round: 1 if this m is a win against Aoki's move.
            cur_score = 1 if m == (a + 1) % 3 else 0
            best_prev = NEG_INF
            # Takahashi's consecutive move must be different.
            # So consider previous moves that are not equal to the current move m.
            for prev in range(3):
                if prev == m:
                    continue
                if dp[prev] > best_prev:
                    best_prev = dp[prev]
            if best_prev != NEG_INF:
                new_dp[m] = best_prev + cur_score
        dp = new_dp
    
    # The answer is the maximum wins Takahashi can achieve over all possible moves in the final round.
    ans = max(dp)
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()