MOD = 998244353

def mod_pow(x, n):
    if n == 0:
        return 1
    elif n % 2 == 1:
        return x * mod_pow(x, n - 1) % MOD
    else:
        y = mod_pow(x, n // 2)
        return y * y % MOD

def solve(N, K):
    if K >= N:
        return (N + 1) // 2
    else:
        inv = mod_pow(N - K, MOD - 2)
        return (N + 1) * mod_pow(2, K, MOD) * inv % MOD

N, K = map(int, input().split())
print(solve(N, K))