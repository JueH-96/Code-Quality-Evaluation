import sys
import math

def prime_factors(n):
    """Returns a dictionary of prime factors and their powers"""
    factors = {}
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors[i] = factors.get(i, 0) + 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def num_divisors(factors):
    """Calculates the number of divisors given prime factors"""
    num = 1
    for power in factors.values():
        num *= (power + 1)
    return num

def calculate_score(N, M):
    """Calculates the sum of scores of all good sequences"""
    MOD = 998244353
    score = 0
    for length in range(1, N + 1):
        for seq in range(M ** length):
            num = 1
            for i in range(length):
                num *= (seq // (M ** i)) % M + 1
            factors = prime_factors(num)
            score += num_divisors(factors)
            score %= MOD
    return score

def main():
    N, M = map(int, sys.stdin.readline().split())
    print(calculate_score(N, M))

if __name__ == "__main__":
    main()