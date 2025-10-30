# YOUR CODE HERE

import sys

MOD = 998244353

def solve():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))

    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        dp[i] = dp[i - 1][:]
        for j in range(i, 0, -1):
            dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD

    ans = 0
    for i in range(1, n + 1):
        if sum(a[:i]) == k:
            ans = (ans + pow(2, n - 1, MOD)) % MOD
        else:
            ans = (ans + dp[n - 1][i - 1]) % MOD

    print(ans)

solve()