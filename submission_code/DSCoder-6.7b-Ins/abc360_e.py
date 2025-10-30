MOD = 998244353
N, K = map(int, input().split())

fact = [1] * (2*N + 1)
for i in range(1, 2*N + 1):
    fact[i] = (fact[i-1] * i) % MOD

invfact = [1] * (2*N + 1)
invfact[2*N] = pow(fact[2*N], MOD - 2, MOD)
for i in range(2*N - 1, -1, -1):
    invfact[i] = (invfact[i+1] * (i + 1)) % MOD

def comb(n, k):
    if k < 0 or k > n:
        return 0
    return (fact[n] * invfact[k] % MOD) * invfact[n-k] % MOD

def solve(n, k):
    if n == 1:
        return 1
    if k == 0:
        return 0
    return (comb(n, 2) * solve(n-2, k-1) % MOD + (n-1) * solve(n-1, k-1) % MOD) % MOD

print(solve(N, K))