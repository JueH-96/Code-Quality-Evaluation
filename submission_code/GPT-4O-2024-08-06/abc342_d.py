import sys
import math
from collections import defaultdict

def prime_factors(n):
    factors = defaultdict(int)
    # Check for number of 2s
    while n % 2 == 0:
        factors[2] += 1
        n //= 2
    # Check for odd factors
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors[i] += 1
            n //= i
    # If n is a prime number greater than 2
    if n > 2:
        factors[n] += 1
    return factors

def factorization_pattern(n):
    factors = prime_factors(n)
    pattern = tuple((p, exp % 2) for p, exp in sorted(factors.items()))
    return pattern

def count_square_pairs(N, A):
    zero_count = A.count(0)
    factorization_count = defaultdict(int)
    
    for num in A:
        if num != 0:
            pattern = factorization_pattern(num)
            factorization_count[pattern] += 1
    
    # Calculate pairs
    total_pairs = 0
    
    # Pairs involving zeros
    total_pairs += zero_count * (N - zero_count) + (zero_count * (zero_count - 1)) // 2
    
    # Pairs from factorization patterns
    for count in factorization_count.values():
        total_pairs += (count * (count - 1)) // 2
    
    return total_pairs

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = count_square_pairs(N, A)
    print(result)