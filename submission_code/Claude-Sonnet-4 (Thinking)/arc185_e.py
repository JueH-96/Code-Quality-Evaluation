import math

MOD = 998244353

n = int(input())
a = list(map(int, input().split()))

# Precompute powers of 2
pow2 = [1] * n
for i in range(1, n):
    pow2[i] = (pow2[i-1] * 2) % MOD

f = 0
for m in range(1, n + 1):
    if m == 1:
        f = 0
    else:
        new_contribution = 0
        for i in range(1, m):  # i from 1 to m-1
            new_contribution = (new_contribution + math.gcd(a[i-1], a[m-1]) * pow2[i-1]) % MOD
        f = (2 * f + new_contribution) % MOD
    print(f)