W, H, L, R, D, U = map(int, input().split())
MOD = 998244353

fact = [1] * (W + H + 1)
for i in range(2, W + H + 1):
    fact[i] = (fact[i - 1] * i) % MOD

inv = [pow(fact[i], MOD - 2, MOD) for i in range(W + H + 1)]

def combinations(n, k):
    if k < 0 or k > n:
        return 0
    return (fact[n] * inv[k] % MOD) * inv[n - k] % MOD

ans = 0
for x in range(W + 1):
    for y in range(H + 1):
        if (x < L or R < x or y < D or U < y):
            ans = (ans + combinations(x + y, x)) % MOD

print(ans)