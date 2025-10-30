import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    P = [0] * (Q+1)
    V = [0] * (Q+1)
    for i in range(1, Q+1):
        P[i], V[i] = map(int, sys.stdin.readline().split())

    MOD = 998244353
    dp = [[0] * (N+1) for _ in range(Q+1)]
    dp[0][0] = 1
    max_v = [0] * (N+1)

    for i in range(1, Q+1):
        for j in range(N+1):
            dp[i][j] = (dp[i-1][j] + dp[i][j-1]) % MOD
            if j >= P[i]:
                dp[i][j] = (dp[i][j] + MOD - dp[i-1][j-P[i]]) % MOD
            if j >= P[i-1]:
                dp[i][j] = (dp[i][j] + MOD - dp[i-1][j-P[i-1]]) % MOD
            if j >= P[i] and V[i] > max_v[j-P[i]]:
                dp[i][j] = (dp[i][j] + 1) % MOD
        max_v[i] = max(max_v[i-1], V[i])

    print(dp[Q][N])

solve()