import sys
from operator import xor
from functools import reduce
from itertools import accumulate

MOD = 998244353
MAX = 200005
fac = [1] * MAX
finv = [1] * MAX
inv = [0, 1] + [0] * (MAX - 2)

for i in range(2, MAX):
    fac[i] = fac[i - 1] * i % MOD
    inv[i] = MOD - inv[MOD % i] * (MOD // i) % MOD
    finv[i] = finv[i - 1] * inv[i] % MOD

def comb(n, k):
    if n < k or n < 0 or k < 0:
        return 0
    return fac[n] * (finv[k] * finv[n - k] % MOD) % MOD

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    pow2 = list(accumulate([1] + [0] * N, lambda x, _: x * 2 % MOD))
    ans = 0
    for i in range(N):
        ans += A[i] * (pow2[i] - comb(i, M - 1)) % MOD
        ans %= MOD
    return ans * pow(K, MOD - 2, MOD) % MOD

print(solve())