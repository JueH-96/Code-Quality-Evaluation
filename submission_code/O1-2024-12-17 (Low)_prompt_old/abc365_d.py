def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Moves that beat a given move
    WIN = {'R': 'P', 'P': 'S', 'S': 'R'}
    # Moves that tie a given move
    TIE = {'R': 'R', 'P': 'P', 'S': 'S'}

    # We'll use DP with two states:
    # dp0[i] = max number of wins up to i-th game if Takahashi plays TIE[S[i]] on i-th game
    # dp1[i] = max number of wins up to i-th game if Takahashi plays WIN[S[i]] on i-th game

    # Since N can be up to 2*10^5, we should use lists (or arrays) for efficiency.
    dp0 = [-10**10] * N  # very large negative for impossible states
    dp1 = [-10**10] * N

    # Initialize for the first game (i=0):
    dp0[0] = 0          # TIE[S[0]] yields 0 wins at round 0
    dp1[0] = 1          # WIN[S[0]] yields 1 win at round 0

    for i in range(1, N):
        # Possible move for the i-th game if Takahashi ties
        move_tie_i = TIE[S[i]]
        # Possible move for the i-th game if Takahashi wins
        move_win_i = WIN[S[i]]

        # The previous tie and win moves
        move_tie_i_1 = TIE[S[i-1]]
        move_win_i_1 = WIN[S[i-1]]

        # dp0[i] transitions (Takahashi tying the i-th game => 0 additional wins):
        # from dp0[i-1] if the moves are different
        if move_tie_i != move_tie_i_1 and dp0[i-1] >= 0:
            dp0[i] = max(dp0[i], dp0[i-1])
        # from dp1[i-1]
        if move_tie_i != move_win_i_1 and dp1[i-1] >= 0:
            dp0[i] = max(dp0[i], dp1[i-1])

        # dp1[i] transitions (Takahashi wins the i-th game => +1 win):
        # from dp0[i-1]
        if move_win_i != move_tie_i_1 and dp0[i-1] >= 0:
            dp1[i] = max(dp1[i], dp0[i-1] + 1)
        # from dp1[i-1]
        if move_win_i != move_win_i_1 and dp1[i-1] >= 0:
            dp1[i] = max(dp1[i], dp1[i-1] + 1)

    # The answer is the maximum of dp0[N-1] and dp1[N-1]
    print(max(dp0[N-1], dp1[N-1]))

# Let's call solve() to execute when this file runs:
# solve()