def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()
    # map each Aoki move to the winning Takahashi move
    win = {'R': 'P', 'P': 'S', 'S': 'R'}
    # Precompute Takahashi's winning move for each position
    W = [win[c] for c in S]

    # dp0 = max wins up to previous position if we chose DRAW there (i.e., T = S[i-1])
    # dp1 = max wins up to previous position if we chose WIN there   (i.e., T = W[i-1])
    dp0 = 0
    dp1 = 1  # at position 0, picking the winning move gives 1 win

    for i in range(1, N):
        s_cur = S[i]
        w_cur = W[i]
        s_pre = S[i-1]
        w_pre = W[i-1]

        # If we draw at i, T[i] = s_cur, no new win
        best_draw = -10**18
        if s_cur != s_pre:
            best_draw = max(best_draw, dp0)
        if s_cur != w_pre:
            best_draw = max(best_draw, dp1)

        # If we win at i, T[i] = w_cur, +1 win
        best_win = -10**18
        if w_cur != s_pre:
            best_win = max(best_win, dp0)
        if w_cur != w_pre:
            best_win = max(best_win, dp1)
        best_win += 1

        dp0, dp1 = best_draw, best_win

    # Edge case N=1: dp0=0, dp1=1 is already correct.
    print(max(dp0, dp1))


if __name__ == "__main__":
    main()