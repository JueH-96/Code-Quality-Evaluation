MOD = 998244353

def solve(n, m):
    if m == 1:
        return n % MOD

    fact = [1] * (m + 1)
    for i in range(2, m + 1):
        x = i
        p = 2
        while p * p <= x:
            if x % p == 0:
                e = 0
                while x % p == 0:
                    x //= p
                    e += 1
                fact[i] *= (2 * e + 1)
                fact[i] %= MOD
            p += 1
        if x > 1:
            fact[i] *= 3
            fact[i] %= MOD

    inv = [1] * (m + 1)
    for i in range(2, m + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    dp = [0] * (m + 1)
    dp[1] = 1
    for i in range(2, m + 1):
        dp[i] = (dp[i - 1] * 2 + fact[i] * inv[i]) % MOD

    ans = 0
    base = 1
    for i in range(1, m + 1):
        ans += base * dp[i]
        ans %= MOD
        base = base * i * inv[m] % MOD
    base = pow(base, n, MOD)
    ans = ans * base % MOD

    return ans

n, m = map(int, input().split())
print(solve(n, m))