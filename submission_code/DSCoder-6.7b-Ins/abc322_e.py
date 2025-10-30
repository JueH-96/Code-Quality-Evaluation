import sys

def solve():
    N, K, P = map(int, sys.stdin.readline().split())
    plans = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

    dp = [[0] * (P + 1) for _ in range(K + 1)]
    for i in range(1, K + 1):
        dp[i][0] = float('inf')

    for i in range(1, N + 1):
        c, *a = plans[i - 1]
        for j in range(K, 0, -1):
            for p in range(P, -1, -1):
                dp[j][p] = min(dp[j][p], dp[j - 1][max(0, p - a[j - 1])] + c)

    if dp[K][P] == float('inf'):
        return -1
    else:
        return dp[K][P]

print(solve())