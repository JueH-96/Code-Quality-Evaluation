# YOUR CODE HERE

import sys

MOD = 998244353

def solve():
    K = int(input())
    C = list(map(int, input().split()))

    dp = [[0] * (K + 1) for _ in range(27)]
    dp[0][0] = 1

    for i in range(1, 27):
        for j in range(K + 1):
            dp[i][j] = dp[i - 1][j]
            for k in range(1, min(j, C[i - 1]) + 1):
                dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) % MOD

    print(dp[26][K])

solve()