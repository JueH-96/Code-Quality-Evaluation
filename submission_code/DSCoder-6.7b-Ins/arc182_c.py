import sys

def solve(n, m):
    MOD = 998244353
    fact = [1] * (n + 1)
    inv = [1] * (n + 1)
    inv_fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % MOD
        inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD
        inv_fact[i] = (inv_fact[i - 1] * inv[i]) % MOD
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, m):
        for j in range(n, 0, -1):
            dp[j] = (dp[j] + dp[j - 1]) % MOD
    ans = 0
    for i in range(1, n + 1):
        ans = (ans + dp[i] * fact[i] % MOD * inv_fact[i] % MOD * pow(i, n, MOD)) % MOD
    return ans

n, m = map(int, sys.stdin.readline().split())
print(solve(n, m))