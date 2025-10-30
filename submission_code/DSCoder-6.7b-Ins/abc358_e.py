import sys

MOD = 998244353

def solve(K, C):
    dp = [[0] * (K + 1) for _ in range(27)]
    dp[0][0] = 1
    for i in range(1, 27):
        dp[i][0] = 1
        for j in range(1, K + 1):
            dp[i][j] = (dp[i][j - 1] + dp[i - 1][j - 1] * (j if C[i - 1] >= j else C[i - 1])) % MOD
    return sum(dp[26][j] for j in range(K + 1)) % MOD

K = int(sys.stdin.readline())
C = list(map(int, sys.stdin.readline().split()))
print(solve(K, C))