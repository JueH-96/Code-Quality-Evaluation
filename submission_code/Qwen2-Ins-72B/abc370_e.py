import sys
from collections import deque

MOD = 998244353

def modpow(x, n):
    if n == 0:
        return 1
    elif n % 2 == 1:
        return (x * modpow(x, n - 1)) % MOD
    else:
        y = modpow(x, n // 2)
        return (y * y) % MOD

def modinv(x):
    return modpow(x, MOD - 2)

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Precompute modular inverses for 1 to N
    invs = [1] * (N + 1)
    invs[1] = modinv(1)
    for i in range(2, N + 1):
        invs[i] = (MOD - MOD // i) * invs[MOD % i] % MOD

    # Precompute binomial coefficients
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    def binom(n, k):
        return fact[n] * invs[k] * invs[n - k] % MOD

    # Compute prefix sums
    prefix_sums = [0]
    for a in A:
        prefix_sums.append(prefix_sums[-1] + a)

    # Compute the number of ways to divide the array without any subsequence summing to K
    dp = [deque() for _ in range(N + 1)]
    dp[0].append(1)
    for i in range(1, N + 1):
        for j in range(i):
            if prefix_sums[i] - prefix_sums[j] == K:
                dp[i].append(0)
            else:
                dp[i].append(sum(dp[j]) % MOD)

        for j in range(i):
            dp[i][j] = (dp[i][j] * binom(i - j - 1, i - j)) % MOD

    print(sum(dp[N]) % MOD)

solve()