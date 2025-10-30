# YOUR CODE HERE
import sys
import math

def solve():
    """
    Solves the 400 Number problem by precomputing necessary data and then
    answering queries in O(1) time per query.

    A positive integer N is a 400 number if it satisfies two conditions:
    1. N has exactly 2 distinct prime factors.
    2. The exponent of each prime factor in the prime factorization of N is a positive even integer.

    This implies N must be a perfect square, N = K^2, where its square root K
    is a number with exactly two distinct prime factors. For example, if K = p^a * q^b
    (with p, q distinct primes and a, b >= 1), then N = K^2 = p^(2a) * q^(2b). This N
    has two distinct prime factors (p, q) and their exponents (2a, 2b) are even and positive.

    The problem asks for the largest 400 number N <= A. This is equivalent to finding
    the largest integer K such that K^2 <= A (i.e., K <= floor(sqrt(A))) and K has
    exactly two distinct prime factors.

    The strategy is to precompute data for all possible values of K. Since A <= 10^12,
    the maximum value of K we need to consider is floor(sqrt(10^12)) = 10^6.
    """

    # --- Precomputation ---

    MAX_K = 1000000

    # Step 1: Use a sieve to find the number of distinct prime factors for all integers up to MAX_K.
    # num_distinct_prime_factors[i] will store the number of distinct prime factors of i.
    num_distinct_prime_factors = [0] * (MAX_K + 1)
    
    for i in range(2, MAX_K + 1):
        if num_distinct_prime_factors[i] == 0:  # This means i is a prime number.
            # Iterate through all multiples of the prime i and increment their count of distinct prime factors.
            for j in range(i, MAX_K + 1, i):
                num_distinct_prime_factors[j] += 1

    # Step 2: Create a lookup table `largest_k_lookup`.
    # largest_k_lookup[i] stores the largest integer j <= i that has exactly two distinct prime factors.
    # This can be computed efficiently by iterating once.
    largest_k_lookup = [0] * (MAX_K + 1)
    for i in range(1, MAX_K + 1):
        if num_distinct_prime_factors[i] == 2:
            largest_k_lookup[i] = i
        else:
            # If i doesn't have 2 prime factors, the largest k up to i is the same as up to i-1.
            largest_k_lookup[i] = largest_k_lookup[i - 1]

    # --- Query Processing ---

    try:
        # Read the number of queries.
        q_str = sys.stdin.readline()
        if not q_str: return
        q = int(q_str)
    except (IOError, ValueError):
        # Handle cases with no input.
        return

    # Use a list to buffer the output lines for faster printing.
    output_buffer = []
    for _ in range(q):
        line = sys.stdin.readline()
        if not line:
            break
        a = int(line)

        # Calculate S = floor(sqrt(A)).
        # math.isqrt() is precise and efficient for this (available in Python 3.8+).
        # We can fallback to int(a**0.5) for older Python versions, which is safe
        # for A <= 10^12 due to the precision of 64-bit floats.
        try:
            s = math.isqrt(a)
        except AttributeError:
            s = int(a**0.5)

        # Using the precomputed table, find the largest K <= S with two distinct prime factors.
        k = largest_k_lookup[s]

        # The answer is K^2.
        ans = k * k
        output_buffer.append(str(ans))

    # Print all results at once, separated by newlines.
    if output_buffer:
        sys.stdout.write('
'.join(output_buffer) + '
')

# The script is executed once, so we call solve() directly.
if __name__ == "__main__":
    solve()