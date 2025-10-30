# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    N, Q, *PV = map(int, input().split())
    mod = 998244353

    P = PV[::2]
    V = PV[1::2]

    dp = [[0] * (N + 1) for _ in range(Q + 1)]
    dp[0][0] = 1

    for i in range(Q):
        p, v = P[i], V[i]
        for j in range(N + 1):
            if dp[i][j] == 0:
                continue
            if j < p and v >= V[i - 1] if i > 0 else True:
                dp[i + 1][p] = (dp[i + 1][p] + dp[i][j]) % mod
            if v >= V[i - 1] if i > 0 else True:
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % mod
            if j >= p and v >= V[i - 1] if i > 0 else True:
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % mod

    print(sum(dp[Q]) % mod)

solve()