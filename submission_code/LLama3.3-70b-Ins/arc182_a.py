import sys
input = sys.stdin.readline

MOD = 998244353

def solve():
    N, Q = map(int, input().split())
    P = []
    V = []
    for _ in range(Q):
        p, v = map(int, input().split())
        P.append(p)
        V.append(v)

    dp = [[0, 0] for _ in range(Q)]
    dp[0][0] = 1
    dp[0][1] = 1

    for i in range(1, Q):
        for j in range(2):
            for k in range(2):
                if j == 0 and P[i-1] >= P[i] and V[i-1] > V[i]:
                    continue
                if j == 1 and P[i-1] <= P[i] and V[i-1] > V[i]:
                    continue
                dp[i][k] = (dp[i][k] + dp[i-1][j]) % MOD

    print((dp[Q-1][0] + dp[Q-1][1]) % MOD)

solve()