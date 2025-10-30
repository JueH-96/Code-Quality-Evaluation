MOD = 998244353

def prime_factors(n):
    factors = {}
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
       factors[n] = factors.get(n, 0) + 1
    return factors

def mod_pow(base, exponent, modulus):
    result = 1
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

def sum_of_scores(N, M):
    total_score = 0
    for i in range(1, M + 1):
        factors = prime_factors(i)
        divisors = 1
        for exponent in factors.values():
            divisors *= (exponent + 1)
        total_score += divisors * (mod_pow(M, N, MOD) - mod_pow(M - 1, N, MOD))
        total_score %= MOD
    return total_score

N, M = map(int, input().split())
print(sum_of_scores(N, M))