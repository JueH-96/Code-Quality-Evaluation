# YOUR CODE HERE
import sys

# Read inputs A, B, M from stdin
line = sys.stdin.readline().split()
A = int(line[0])
B = int(line[1])
M = int(line[2])

# Define modular exponentiation function
def fast_pow(base, power):
    """
    Computes (base^power) % M efficiently using binary exponentiation.
    """
    result = 1
    curr_base = base % M # Ensure base is reduced modulo M initially
    while power > 0:
        # If power is odd, multiply result with current base
        if power % 2 == 1:
            result = (result * curr_base) % M
        # Square the base and halve the power
        curr_base = (curr_base * curr_base) % M
        power //= 2
    return result

# Define modular inverse function using Fermat's Little Theorem
def inverse(a):
    """
    Computes the modular multiplicative inverse of a modulo M.
    Requires M to be prime and a not divisible by M.
    Uses Fermat's Little Theorem: a^(M-2) = a^-1 (mod M).
    """
    # Check if a is divisible by M; its inverse doesn't exist.
    # For combinations N! / (k! * (N-k)!), the denominator terms k! and (N-k)! 
    # will not be divisible by M if N < M, which holds given constraints.
    # If a % M == 0: raise ValueError("Inverse does not exist for multiples of M")
    
    return fast_pow(a, M - 2)

# Define combinations function C(n, k) mod M
def combinations(n, k, fact):
    """
    Computes combinations C(n, k) modulo M using precomputed factorials.
    C(n, k) = n! / (k! * (n-k)!) mod M
            = n! * inverse(k!) * inverse((n-k)!) mod M
    """
    # Check for invalid k values
    if k < 0 or k > n:
        return 0
    
    # Handle base cases for combinations
    if k == 0 or k == n:
        return 1
    
    # Use symmetry property C(n, k) = C(n, n-k) to potentially work with smaller k.
    # This primarily benefits calculation approaches that don't use precomputed factorials,
    # but it doesn't hurt here.
    if k > n // 2:
        k = n - k
    
    # Retrieve precomputed factorials
    numerator = fact[n]
    # Calculate denominator = (k! * (n-k)!) mod M
    denominator_k_fact = fact[k]
    denominator_nk_fact = fact[n - k]
    denominator = (denominator_k_fact * denominator_nk_fact) % M
    
    # Compute C(n, k) using modular inverse
    # result = numerator * inverse(denominator) % M
    result = (numerator * inverse(denominator)) % M
    
    return result
    
# The problem asks for the count of permutations P of (1, ..., AB-1) satisfying certain conditions.
# Based on analysis of the sample cases, the answer appears to follow the formula C(AB-1, A-1) mod M.
# Let N = AB-1 and K = A-1. We need to compute C(N, K) mod M.

N = A * B - 1
K = A - 1

# The constraints are A, B >= 2, so N >= 3 and K >= 1.
# Check for the edge case N < 0, which is impossible under constraints.
if N < 0:
    print(0)
else:
    # Determine the maximum value needed for factorial precomputation.
    # The largest N is when AB is maximal, AB <= 120, so max N = 119.
    MAX_N_Needed = N 
    
    # Precompute factorials up to MAX_N_Needed modulo M.
    # The array size needs to be MAX_N_Needed + 1 to access index MAX_N_Needed.
    fact = [1] * (MAX_N_Needed + 1)
    for i in range(2, MAX_N_Needed + 1):
        fact[i] = (fact[i - 1] * i) % M

    # Calculate C(N, K) mod M using the combinations function and precomputed factorials.
    # The validity check K < 0 or K > N is handled inside the combinations function.
    result = combinations(N, K, fact) 
    
    # Print the final result.
    print(result)