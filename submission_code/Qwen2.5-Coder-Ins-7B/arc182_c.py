import sys
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

def divisors_count(factors):
    count = 1
    for exp in factors.values():
        count *= (exp + 1)
        count %= MOD
    return count

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])

    total_score = 0
    for i in range(1, M + 1):
        factors = prime_factors(i)
        total_score += divisors_count(factors)
        total_score %= MOD

    print(total_score)

if __name__ == "__main__":
    main()