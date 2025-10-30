from math import sqrt, isqrt
from collections import defaultdict

def prime_factorize(n):
    i = 2
    factors = defaultdict(int)
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors[i] += 2
    if n > 1:
        factors[n] += 2
    return factors

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Prime factorization for each A[i]
    pf = [prime_factorize(a) for a in A]

    # Count the number of occurrences of each prime factor
    factors_count = defaultdict(int)
    for pfd in pf:
        for p, e in pfd.items():
            factors_count[p] += e

    # Count the number of square pairs
    square_pairs = 0
    for i in range(N):
        for p, e in pf[i].items():
            factors_count[p] -= e
        is_square = 1
        for e in factors_count.values():
            if e % 2:
                is_square = 0
                break
        square_pairs += is_square * (N - 1 - i)
        for p, e in pf[i].items():
            factors_count[p] += e

    print(square_pairs)

solve()