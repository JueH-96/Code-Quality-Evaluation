# YOUR CODE HERE
import sys
import math
from collections import Counter

# Set the maximum possible value for A_i + 1
# Constraints state 0 <= A_i <= 2 * 10^5
MAX_VAL = 200001 

# Precompute the smallest prime factor (spf) for numbers up to MAX_VAL
# spf[i] will store the smallest prime factor of i
spf = list(range(MAX_VAL))

def sieve():
    """ Precomputes smallest prime factors using a sieve method. """
    # We only need to check for primes up to sqrt(MAX_VAL)
    # because any composite number n must have a prime factor <= sqrt(n).
    limit = int(math.sqrt(MAX_VAL)) + 1
    for i in range(2, limit):
        # If i is prime (its smallest prime factor is itself)
        if spf[i] == i: 
            # Mark multiples of i starting from i*i.
            # Multiples smaller than i*i would have already been marked
            # by primes smaller than i.
            for j in range(i * i, MAX_VAL, i):
                # Update spf[j] only if it hasn't been marked by a smaller prime factor yet.
                # This ensures spf[j] correctly stores the *smallest* prime factor.
                if spf[j] == j:
                    spf[j] = i

# Run the sieve function once at the start to populate the spf array
sieve() 

def get_square_free(n):
    """
    Calculates the square-free part of a positive integer n using the precomputed spf array.
    The square-free part of a number is obtained by multiplying the prime factors 
    that appear an odd number of times in its prime factorization.
    Example: 12 = 2^2 * 3^1 -> square-free part is 3.
             72 = 2^3 * 3^2 -> square-free part is 2.
    This function assumes n > 0.
    """
    # Base case: square-free part of 1 is 1.
    if n == 1:
        return 1
    
    res = 1 # Initialize the square-free part result
    
    # Factorize n using the precomputed smallest prime factors
    while n > 1:
        p = spf[n] # Get the smallest prime factor of the current n
        count = 0
        
        # Divide n by p repeatedly and count the occurrences of this prime factor p
        while n % p == 0:
            count += 1
            n //= p
            
        # If the prime factor p occurred an odd number of times in the factorization,
        # include it once in the square-free result.
        if count % 2 == 1:
            res *= p
            
    # The loop terminates when n becomes 1 after all prime factors are processed.
    return res

def solve():
    """ Reads input, solves the problem, and prints the output. """
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    num_zeros = 0
    
    # Use collections.Counter to efficiently store frequencies of 
    # the square-free parts of the positive numbers in array A.
    sf_counts = Counter()
    
    # Process each element in the input array A
    for x in a:
        if x == 0:
            # Count the number of zeros
            num_zeros += 1
        elif x > 0:
            # For positive numbers, calculate their square-free part
            sf = get_square_free(x)
            # Increment the count for this specific square-free part
            sf_counts[sf] += 1

    # Calculate the number of pairs (i, j) with i < j where at least one of A_i, A_j is zero.
    # If either A_i or A_j is 0, their product A_i * A_j = 0, which is a square number (0^2).
    # The number of such pairs can be calculated as:
    # Pairs where both A_i=0 and A_j=0: num_zeros * (num_zeros - 1) // 2
    # Pairs where one is 0 and the other is non-zero: num_zeros * (n - num_zeros)
    zero_pairs = num_zeros * (num_zeros - 1) // 2 + num_zeros * (n - num_zeros)

    # Calculate the number of pairs (i, j) with i < j where both A_i > 0 and A_j > 0,
    # and A_i * A_j is a square number.
    # A_i * A_j is a square number if and only if A_i and A_j have the same square-free part.
    positive_pairs = 0
    # Iterate through the counts of each unique square-free part found among positive numbers
    for sf in sf_counts:
        count = sf_counts[sf]
        # If a square-free part `sf` appears `count` times, we can form 
        # `count * (count - 1) // 2` pairs of numbers having this square-free part.
        positive_pairs += count * (count - 1) // 2

    # The total number of pairs satisfying the conditions is the sum of pairs involving zeros
    # and pairs of non-zeros whose product is a square.
    print(zero_pairs + positive_pairs)

# Execute the solver function
solve()