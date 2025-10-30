# YOUR CODE HERE

import sys

def solve():
    N = int(input())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())

    dp = [[[float('inf')] * (K2 + 1) for _ in range(K1 + 1)] for _ in range(N + 1)]
    dp[0][0][0] = 0

    for i in range(N):
        for j in range(K1 + 1):
            for k in range(K2 + 1):
                if dp[i][j][k] == float('inf'):
                    continue
                if j + 1 <= K1:
                    dp[i + 1][j + 1][k] = min(dp[i + 1][j + 1][k], dp[i][j][k] + C1)
                if k + 1 <= K2:
                    dp[i + 1][j][k + 1] = min(dp[i + 1][j][k + 1], dp[i][j][k] + C2)
                dp[i + 1][j][k] = min(dp[i + 1][j][k], dp[i][j][k])

    ans = float('inf')
    for j in range(K1 + 1):
        for k in range(K2 + 1):
            if L1 * j + L2 * k >= D[-1]:
                ans = min(ans, dp[N][j][k])

    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()