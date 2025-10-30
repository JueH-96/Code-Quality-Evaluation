def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S_str = input_data[1]

    # Map R, P, S to 0,1,2
    char2idx = {'R':0, 'P':1, 'S':2}
    S = [char2idx[c] for c in S_str]

    # Winning move for each of Aoki's moves:
    # If Aoki is R(0), T must use P(1) to win.
    # If Aoki is P(1), T must use S(2) to win.
    # If Aoki is S(2), T must use R(0) to win.
    win_move = [1, 2, 0]  

    # Valid non-losing moves for T vs each Aoki's move:
    # If Aoki is R(0): T can use R(0) or P(1).
    # If Aoki is P(1): T can use P(1) or S(2).
    # If Aoki is S(2): T can use S(2) or R(0).
    valid_moves = [
        [0, 1],  # vs R
        [1, 2],  # vs P
        [2, 0],  # vs S
    ]

    # dp[i][m] = maximum number of wins up to (and including) game i 
    #            if Takahashi's move for game i is m.
    # We'll only fill dp for m in valid_moves[S[i]].
    # Also T[i] != T[i-1].
    # dp arrays for prev (i-1) and cur (i).
    prev_dp = [-1]*3
    cur_dp  = [-1]*3

    # Initialize for i=0
    first_valid = valid_moves[S[0]]
    for m in first_valid:
        # Win if m == win_move[S[0]]
        prev_dp[m] = 1 if m == win_move[S[0]] else 0

    # Fill dp for i = 1..N-1
    for i in range(1, N):
        # Reset cur_dp
        cur_dp = [-1]*3
        vm = valid_moves[S[i]]
        for m in vm:
            # We want dp[i][m] = max over m' != m of dp[i-1][m'] + (1 if m is winning).
            best_prev = -1
            for m_prev in range(3):
                if m_prev != m and prev_dp[m_prev] != -1:
                    if prev_dp[m_prev] > best_prev:
                        best_prev = prev_dp[m_prev]
            if best_prev != -1:
                cur_dp[m] = best_prev + (1 if m == win_move[S[i]] else 0)
        prev_dp, cur_dp = cur_dp, prev_dp  # swap references

    # The answer is the maximum over dp of the last step (prev_dp)
    print(max(prev_dp))

# Do not forget to call main()!
if __name__ == "__main__":
    main()