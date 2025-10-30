def count_permutations(A, B, M):
    from math import factorial
    
    # Calculate the number of permutations using the formula:
    # The number of permutations of (1, 2, ..., AB-1) with the given constraints
    # is given by the formula: (AB-1)! / (A-1)! / (B-1)!
    
    # Calculate factorials and their modular inverses
    def mod_inv(x, mod):
        return pow(x, mod - 2, mod)
    
    # Precompute factorials and their inverses
    max_n = A * B - 1
    fact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = fact[i - 1] * i % M
    
    # Calculate the result using the formula
    numerator = fact[max_n]
    denominator = (fact[A - 1] * fact[B - 1]) % M
    result = (numerator * mod_inv(denominator, M)) % M
    
    return result

# Read input
import sys
input = sys.stdin.read
A, B, M = map(int, input().strip().split())

# Calculate and print the result
print(count_permutations(A, B, M))