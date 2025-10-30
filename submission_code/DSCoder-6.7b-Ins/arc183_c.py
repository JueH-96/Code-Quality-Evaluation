import sys

MOD = 998244353
N, M = map(int, sys.stdin.readline().split())
conditions = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]

fact = [1] * (N + 1)
for i in range(1, N + 1):
    fact[i] = (fact[i - 1] * i) % MOD

inv_fact = [1] * (N + 1)
inv_fact[-1] = pow(fact[-1], MOD - 2, MOD)
for i in range(N - 1, -1, -1):
    inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

def comb(n, k):
    if n < k or k < 0:
        return 0
    return (fact[n] * inv_fact[k] % MOD) * inv_fact[n - k] % MOD

count = 1
for L, R, X in conditions:
    count = (count * comb(R - L + 1, X - L)) % MOD

sys.stdout.write(str(count))