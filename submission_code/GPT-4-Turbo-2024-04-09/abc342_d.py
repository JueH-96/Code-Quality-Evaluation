import sys
import math
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

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

def normalize_factors(factors):
    normalized = {}
    for p, exp in factors.items():
        if exp % 2 != 0:
            normalized[p] = 1
    return tuple(sorted(normalized.items()))

# Dictionary to count normalized factor occurrences
factor_count = defaultdict(int)

for a in A:
    if a == 0:
        factors = (0,)
    else:
        factors = prime_factors(a)
        factors = normalize_factors(factors)
    factor_count[factors] += 1

result = 0
for count in factor_count.values():
    if count > 1:
        result += count * (count - 1) // 2

print(result)