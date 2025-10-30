# YOUR CODE HERE
MOD = 998244353

def prime_factors(n):
    i = 2
    factors = {}
    while i * i <= n:
        while (n % i) == 0:
            if i in factors:
                factors[i] += 1
            else:
                factors[i] = 1
            n //= i
        i += 1
    if n > 1:
        factors[n] = 1
    return factors

def count_divisors(factors):
    divisors = 1
    for exponent in factors.values():
        divisors *= (exponent + 1)
    return divisors

def sum_of_scores(N, M):
    total_score = 0
    for product in range(1, M + 1):
        factors = prime_factors(product)
        divisors = count_divisors(factors)
        total_score = (total_score + divisors * pow(M, N - 1, MOD)) % MOD
    return total_score

import sys
input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
print(sum_of_scores(N, M))