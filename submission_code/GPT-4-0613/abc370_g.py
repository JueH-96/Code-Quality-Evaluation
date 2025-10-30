import sys
from operator import mul
from functools import reduce
from collections import defaultdict

MOD = 998244353
N, M = map(int, sys.stdin.readline().split())
MAX = int(N ** 0.5) + 1

# Factorization
def factorization(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n ** 0.5 // 1)) + 1):
        if temp % i == 0:
            cnt = 0
            while temp % i == 0:
                cnt += 1
                temp //= i
            arr.append([i, cnt])
    if temp != 1:
        arr.append([temp, 1])
    if arr == []:
        arr.append([n, 1])
    return arr

# Combination
def comb(n, r, mod=MOD):
    if r < 0 or r > n:
        return 0
    return fact[n] * factinv[r] % mod * factinv[n - r] % mod

# Preparation
fact = [1, 1]
factinv = [1, 1]
inv = [0, 1]
for i in range(2, M + MAX + 5):
    fact.append((fact[-1] * i) % MOD)
    inv.append((-inv[MOD % i] * (MOD // i)) % MOD)
    factinv.append((factinv[-1] * inv[-1]) % MOD)

# Count the number of divisors
div = defaultdict(int)
for i in range(1, MAX):
    if i * i > N:
        break
    if N % i == 0:
        div[i] += 1
        if i * i != N:
            div[N // i] += 1

# Count the number of divisors of 2 and 3
div2 = defaultdict(int)
div3 = defaultdict(int)
for d, cnt in div.items():
    f = factorization(d)
    for base, exponent in f:
        if base == 2:
            div2[exponent] += cnt
        if base == 3:
            div3[exponent] += cnt

# Count the number of sequences
ans = 0
for i in range(MAX):
    if i * 2 > M:
        break
    for j in range(MAX):
        if i * 2 + j > M or i + j * 2 > M:
            break
        tmp = comb(M, i) * comb(M - i, j) % MOD
        tmp *= pow(div2[i], M - i - j, MOD) * pow(div3[j], M - i - j, MOD) % MOD
        tmp *= pow(div[1] - div2[i] - div3[j], M - i - j, MOD) % MOD
        ans += tmp
        ans %= MOD
print(ans)