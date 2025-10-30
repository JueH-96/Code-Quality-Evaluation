from collections import defaultdict
from math import gcd
from sys import stdin

def prime_factors(n):
    factors = defaultdict(int)
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] += 1
            n //= d
        d += 1
    if n > 1:
        factors[n] += 1
    return factors

def normalize_factors(factors):
    return {p: exp % 2 for p, exp in factors.items() if exp % 2 == 1}

def main():
    input_lines = stdin.read().splitlines()
    N = int(input_lines[0])
    A = list(map(int, input_lines[1].split()))

    normalized_factors = []
    for a in A:
        factors = prime_factors(a)
        normalized = normalize_factors(factors)
        normalized_factors.append(frozenset(normalized.items()))

    count = defaultdict(int)
    for factors in normalized_factors:
        count[factors] += 1

    answer = 0
    for factors in normalized_factors:
        answer += count[factors] - 1
        count[factors] -= 1

    print(answer)

if __name__ == "__main__":
    main()