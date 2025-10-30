# YOUR CODE HERE

import sys

MOD = 998244353

def solve():
    n, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()

    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(n):
        for j in range(i + 1):
            dp[i + 1][j + 1] += dp[i][j]
            dp[i + 1][j + 1] %= MOD
            if j >= k:
                dp[i + 1][j - k + 1] += dp[i][j]
                dp[i + 1][j - k + 1] %= MOD

    ans = 1
    for i in range(n):
        if s[i] == 'A':
            ans *= dp[n - i - 1][0]
            ans %= MOD
        elif s[i] == 'B':
            ans *= dp[n - i - 1][1]
            ans %= MOD
        else:
            ans *= (dp[n - i - 1][0] + dp[n - i - 1][1])
            ans %= MOD

    print(ans)

solve()