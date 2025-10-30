N, M = map(int, input().split())
MOD = 998244353

def solve(N, M):
    fact = [1] * (N + 1)
    inv = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
        inv[i] = pow(fact[i], MOD - 2, MOD)

    def C(n, k):
        if k > n or k < 0:
            return 0
        return (fact[n] * inv[k] % MOD) * inv[n - k] % MOD

    res = 0
    for k in range(1, N + 1):
        res = (res + C(N, k) * M ** k) % MOD
    return res

print(solve(N, M))