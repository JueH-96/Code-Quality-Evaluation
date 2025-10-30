def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * (b // gcd(a, b))

def f(x):
    from math import gcd
    from fractions import Fraction
    p, q = Fraction(x).as_integer_ratio()
    return p * q

def find_good_sequences(N, A):
    from collections import defaultdict
    from itertools import product
    
    MOD = 998244353
    
    # Generate all possible pairs (P, Q) such that P * Q = A_i
    pairs = defaultdict(list)
    
    for i in range(N - 1):
        a_i = A[i]
        for P in range(1, int(a_i**0.5) + 1):
            if a_i % P == 0:
                Q = a_i // P
                pairs[i].append((P, Q))
    
    total_sum = 0
    
    # Generate all combinations of (P, Q) for the sequence
    for combination in product(*pairs.values()):
        S = [0] * N
        for i in range(N - 1):
            P, Q = combination[i]
            S[i] = P
            S[i + 1] = Q
        
        # Check if gcd(S) == 1
        if gcd(S[0], S[1]) == 1:
            for i in range(2, N):
                if gcd(S[0], S[i]) != 1:
                    break
            else:
                # Calculate the score
                score = 1
                for num in S:
                    score = (score * num) % MOD
                total_sum = (total_sum + score) % MOD
    
    return total_sum

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the result
result = find_good_sequences(N, A)

# Print the result
print(result)