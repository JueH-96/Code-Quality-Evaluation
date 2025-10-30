MOD = 998244353
N, K = map(int, input().split())
fact = [1] * (K+2)
inv = [1] * (K+2)
for i in range(1, K+2):
    fact[i] = fact[i-1] * i % MOD
    inv[i] = pow(fact[i], MOD-2, MOD)

def C(n, r):
    if r < 0 or r > n:
        return 0
    return fact[n] * inv[r] * inv[n-r] % MOD

ans = 0
for i in range(1, min(N, K+1)):
    ans = (ans + C(N, i) * C(N-1, i-1)) % MOD
print((pow(N, K, MOD) - ans + MOD) % MOD)