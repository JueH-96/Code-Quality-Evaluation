import sys

def solve(n, m):
    MOD = 998244353
    divisors = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            divisors[j] += i
    dp = [0] * (n + 1)
    dp[0] = 1
    for _ in range(m):
        new_dp = dp[:]
        for i in range(n, -1, -1):
            new_dp[i] = (new_dp[i] + dp[i]) % MOD
            if i > 0:
                new_dp[i] = (new_dp[i] + dp[i - 1] * divisors[i]) % MOD
        dp = new_dp
    return dp[n]

n, m = map(int, sys.stdin.readline().split())
print(solve(n, m))