import sys

MOD = 998244353

def mod_inv(a, m):
    def ext_gcd(a, b):
        if a == 0:
            return b, 0, 1
        else:
            g, x, y = ext_gcd(b % a, a)
            return g, y - (b // a) * x, x
    g, x, _ = ext_gcd(a, m)
    return (x % m + m) % m

def solve():
    N, K = map(int, sys.stdin.readline().split())
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact = [1] * (N + 1)
    inv_fact[N] = mod_inv(fact[N], MOD)
    for i in range(N, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % MOD
    dp = [0] * (N + 1)
    dp[0] = 1
    for _ in range(K):
        ndp = [0] * (N + 1)
        for i in range(N + 1):
            for j in range(i + 1):
                ndp[j] = (ndp[j] + dp[i] * fact[i] % MOD * inv_fact[j] % MOD * inv_fact[i - j] % MOD) % MOD
        dp = ndp
    ans = 0
    for i in range(N + 1):
        ans = (ans + dp[i] * fact[N] % MOD * inv_fact[i] % MOD * inv_fact[N - i] % MOD * i % MOD) % MOD
    print(ans)

solve()