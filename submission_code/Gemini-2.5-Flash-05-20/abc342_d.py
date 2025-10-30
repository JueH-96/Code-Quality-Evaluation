import sys
from collections import defaultdict

# Max value for A_i as per constraints
MAX_A_VAL = 2 * 10**5

# Precompute primes up to sqrt(MAX_A_VAL) using Sieve of Eratosthenes
# The square-free part calculation will iterate through these primes.
primes = []
# Calculate limit for sieve: sqrt(MAX_A_VAL)
# For 2*10^5, sqrt is approximately 447.2. So, we need primes up to 447.
sieve_limit = int(MAX_A_VAL**0.5) + 1 
is_prime = [True] * sieve_limit

# 0 and 1 are not prime numbers
if sieve_limit > 0:
    is_prime[0] = False
if sieve_limit > 1:
    is_prime[1] = False

for p in range(2, sieve_limit):
    if is_prime[p]:
        primes.append(p)
        # Mark multiples as not prime
        for multiple in range(p * p, sieve_limit, p):
            is_prime[multiple] = False

# Function to calculate the square-free part of a positive integer n.
# For a number n = p1^e1 * p2^e2 * ... * pk^ek,
# its square-free part is p1^(e1%2) * p2^(e2%2) * ... * pk^(ek%2).
def get_square_free_part(n):
    sfp = 1
    temp_n = n # Use a temporary variable to avoid modifying n

    # Iterate through precomputed primes
    for p in primes:
        # Optimization: If p*p is greater than the remaining temp_n,
        # then temp_n must be either 1 or a prime number itself.
        # Any prime factor of temp_n at this point must be greater than sqrt(original n),
        # meaning it can only appear once (odd power) as a factor.
        if p * p > temp_n:
            break
        
        # If p is a factor of temp_n
        if temp_n % p == 0:
            count = 0
            # Count occurrences of p and remove all factors of p from temp_n
            while temp_n % p == 0:
                count += 1
                temp_n //= p
            # If the count of p is odd, include p in the square-free part
            if count % 2 == 1: # (exponent % 2 == 1)
                sfp *= p
    
    # If temp_n is still greater than 1 after dividing by all primes up to sqrt(original n),
    # it means the remaining temp_n is a prime factor itself (and it must appear with an odd exponent, which is 1).
    if temp_n > 1:
        sfp *= temp_n
        
    return sfp

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())
    # Read the array A from standard input
    A = list(map(int, sys.stdin.readline().split()))

    # Dictionary to store counts of each square-free part for positive numbers
    sq_free_counts = defaultdict(int)
    # Counter for zero values in the input array
    num_zeros = 0

    # Process each number in the input array A
    for x in A:
        if x == 0:
            num_zeros += 1
        else:
            # Calculate square-free part for positive numbers
            sfp = get_square_free_part(x)
            sq_free_counts[sfp] += 1

    total_pairs = 0

    # Calculate pairs where both A_i, A_j are positive
    # A_i * A_j is a square if and only if square_free_part(A_i) == square_free_part(A_j).
    # For each square-free part value, if it appears `count` times, then `count * (count - 1) / 2` pairs can be formed
    # among elements sharing that square-free part.
    for sfp_val in sq_free_counts:
        count = sq_free_counts[sfp_val]
        total_pairs += count * (count - 1) // 2

    # Calculate pairs involving at least one zero
    # If A_i=0 or A_j=0, then A_i*A_j = 0, which is a square (0 = 0^2).
    # These pairs are counted separately to avoid issues with `get_square_free_part(0)` definition
    # and because any pair with a zero is valid regardless of the other number's value.
    
    # 1. Pairs where both A_i and A_j are zero: num_zeros C 2
    total_pairs += num_zeros * (num_zeros - 1) // 2
    
    # 2. Pairs where one element is zero and the other is a non-zero positive number:
    # There are `num_zeros` zeros and `(N - num_zeros)` non-zero numbers.
    total_pairs += num_zeros * (N - num_zeros)

    # Print the final calculated total number of pairs
    sys.stdout.write(str(total_pairs) + "
")

# Call the solve function when the script is executed
if __name__ == '__main__':
    solve()