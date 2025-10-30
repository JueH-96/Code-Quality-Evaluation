# YOUR CODE HERE
from math import sqrt
from collections import defaultdict

MOD = 998244353

def prime_factors(n):
    i = 2
    factors = defaultdict(int)
    while i * i <= n:
        while (n % i) == 0:
            factors[i] += 1
            n //= i
        i += 1
    if n > 1:
        factors[n] += 1
    return factors

def score(n):
    factors = prime_factors(n)
    score = 1
    for exp in factors.values():
        score *= (exp + 1)
    return score % MOD

def solve(N, M):
    dp = [0] * (M + 1)
    dp[0] = 1
    for _ in range(N):
        new_dp = [0] * (M + 1)
        for i in range(M + 1):
            for j in range(M + 1):
                new_dp[i] += dp[i * j % (M + 1)]
                new_dp[i] %= MOD
        dp = new_dp
    total_score = 0
    for i in range(1, M + 1):
        total_score += dp[i] * score(i)
        total_score %= MOD
    return total_score

N, M = map(int, input().split())
print(solve(N, M))