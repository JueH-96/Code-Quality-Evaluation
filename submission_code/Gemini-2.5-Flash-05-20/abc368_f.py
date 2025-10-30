import sys

# Maximum possible value for A_i
MAX_A = 10**5

# sum_of_exponents[i] will store the sum of exponents in the prime factorization of i.
# For example, sum_of_exponents[12] = 3 (since 12 = 2^2 * 3^1, exponents are 2 and 1, sum = 3).
sum_of_exponents = [0] * (MAX_A + 1)

# lp[i] stores the lowest prime factor of i. Used in the sieve.
lp = [0] * (MAX_A + 1)

# List to store prime numbers found during the sieve process.
primes = []

# Sieve to precompute sum_of_exponents for all numbers up to MAX_A.
# This is a highly optimized linear sieve (or nearly linear).
for i in range(2, MAX_A + 1):
    if lp[i] == 0:  # If lp[i] is 0, it means i is a prime number.
        lp[i] = i
        primes.append(i)
        sum_of_exponents[i] = 1  # For a prime number p, its prime factorization is p^1, so sum of exponents is 1.
    
    # Iterate through the primes list. For each prime p, mark its multiples (i*p).
    # We only consider primes p that are less than or equal to lp[i] (the smallest prime factor of i).
    # This ensures that each composite number i*p is processed exactly once, when i = (i*p)/p and p = lp[i*p].
    for p in primes:
        # If p is greater than the smallest prime factor of i, or if i*p exceeds MAX_A,
        # we can stop for this i because any larger prime will also be > lp[i] or cause overflow.
        if p > lp[i] or i * p > MAX_A:
            break
        
        lp[i * p] = p  # The smallest prime factor of i*p is p.
        
        # Calculate sum_of_exponents for i*p based on sum_of_exponents[i].
        # Case 1: p is the smallest prime factor of i (i.e., p == lp[i]).
        # This means that 'p' is already a factor of 'i', and multiplying 'i' by 'p'
        # just increases the exponent of 'p' in the prime factorization.
        # Example: i=4 (2^2), p=2. i*p=8 (2^3). sum_of_exponents[8] = sum_of_exponents[4] + 1 = 2+1=3.
        if p == lp[i]:
            sum_of_exponents[i * p] = sum_of_exponents[i] + 1
        # Case 2: p is smaller than the smallest prime factor of i (i.e., p < lp[i]).
        # This means 'p' is a new prime factor for i*p that was not present in 'i'.
        # Example: i=6 (2^1 * 3^1), p=5. i*p=30 (2^1 * 3^1 * 5^1).
        # sum_of_exponents[30] = sum_of_exponents[6] + 1 = 2+1=3.
        else: # p < lp[i]
            sum_of_exponents[i * p] = sum_of_exponents[i] + 1

# Read N from standard input
N = int(sys.stdin.readline())

# Read the sequence A_i from standard input
A = list(map(int, sys.stdin.readline().split()))

# Calculate the nim-sum (XOR sum of Grundy values)
# The Grundy value for each A_i is sum_of_exponents[A_i].
nim_sum = 0
for a_val in A:
    nim_sum ^= sum_of_exponents[a_val]

# Determine the winner based on the nim-sum
if nim_sum != 0:
    print("Anna")  # If nim-sum is non-zero, the first player (Anna) wins.
else:
    print("Bruno") # If nim-sum is zero, the second player (Bruno) wins.