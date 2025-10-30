import math
from functools import reduce
from fractions import Fraction

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def get_prime_factorization(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def get_coprime_factorizations(n):
    if n == 1:
        return [(1, 1)]
    
    prime_factors = get_prime_factorization(n)
    primes = list(prime_factors.keys())
    
    factorizations = []
    for mask in range(1 << len(primes)):
        p, q = 1, 1
        for i, prime in enumerate(primes):
            if mask & (1 << i):
                p *= prime ** prime_factors[prime]
            else:
                q *= prime ** prime_factors[prime]
        factorizations.append((p, q))
    
    return factorizations

n = int(input())
a = list(map(int, input().split()))

# Get all coprime factorizations for each A_i
factorizations = []
for ai in a:
    factorizations.append(get_coprime_factorizations(ai))

total_score = 0
MOD = 998244353

# Enumerate all combinations
def backtrack(index, current_factorizations):
    global total_score
    
    if index == len(a):
        # Compute the sequence
        ratios = [Fraction(p, q) for p, q in current_factorizations]
        
        # Compute prod_i = r_i * r_{i+1} * ... * r_{N-1}
        prods = []
        for i in range(n - 1):
            prod = Fraction(1)
            for j in range(i, n - 1):
                prod *= ratios[j]
            prods.append(prod)
        
        # Get numerators and denominators
        nums = [prod.numerator for prod in prods]
        dens = [prod.denominator for prod in prods]
        
        # Compute L = lcm of all denominators
        L = 1
        for den in dens:
            L = lcm(L, den)
        
        # Compute the sequence
        sequence = []
        for i in range(n - 1):
            sequence.append(nums[i] * L // dens[i])
        sequence.append(L)
        
        # Check gcd constraint
        seq_gcd = reduce(gcd, sequence)
        if seq_gcd == 1:
            score = 1
            for val in sequence:
                score = (score * val) % MOD
            total_score = (total_score + score) % MOD
        
        return
    
    for factorization in factorizations[index]:
        current_factorizations.append(factorization)
        backtrack(index + 1, current_factorizations)
        current_factorizations.pop()

backtrack(0, [])
print(total_score)