# YOUR CODE HERE

import sys

MOD = 998244353

def solve():
    N, M = map(int, sys.stdin.readline().split())
    conditions = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]

    dp = [[0] * (N + 1) for _ in range(N + 1)]
    dp[0][0] = 1

    for i in range(N):
        for j in range(i + 1):
            dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % MOD
            dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j] * (j + 1)) % MOD

    dp2 = [[0] * (N + 1) for _ in range(N + 1)]
    dp2[0][0] = 1

    for i in range(N):
        for j in range(i + 1):
            dp2[i + 1][j] = (dp2[i + 1][j] + dp2[i][j]) % MOD
            dp2[i + 1][j + 1] = (dp2[i + 1][j + 1] + dp2[i][j] * (i - j + 1)) % MOD

    dp3 = [[0] * (N + 1) for _ in range(N + 1)]
    dp3[0][0] = 1

    for i in range(N):
        for j in range(i + 1):
            dp3[i + 1][j] = (dp3[i + 1][j] + dp3[i][j]) % MOD
            dp3[i + 1][j + 1] = (dp3[i + 1][j + 1] + dp3[i][j] * (j + 1)) % MOD

    for L, R, X in conditions:
        for i in range(L - 1, R):
            for j in range(i + 1):
                dp3[i + 1][j] = (dp3[i + 1][j] - dp[i][j] * dp2[R - i - 1][j]) % MOD
                dp3[i + 1][j + 1] = (dp3[i + 1][j + 1] - dp[i][j] * dp2[R - i - 1][j] * (j + 1)) % MOD

    print(dp3[N][N])

solve()