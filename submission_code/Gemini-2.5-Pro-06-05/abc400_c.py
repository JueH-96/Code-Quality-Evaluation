import sys
import math

def solve():
    """
    Reads a single integer N from standard input and computes the number of 
    "good integers" between 1 and N, inclusive.
    """
    try:
        # Read the input integer N.
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        # Handle potential empty input or invalid format, though problem constraints
        # guarantee a valid integer.
        return

    # A good integer X is of the form 2^a * b^2, where a, b >= 1.
    # We can partition these into two disjoint sets based on the parity of 'a'.

    # Case 1: 'a' is odd.
    # X = 2^(2k+1) * b^2 = 2 * (2^k * b)^2. This is of the form 2 * m^2.
    # We count m >= 1 such that 2 * m^2 <= N, which means m^2 <= N // 2.
    # The number of such m is floor(sqrt(N // 2)).
    count_a_odd = math.isqrt(N // 2)

    # Case 2: 'a' is even.
    # X = 2^(2k) * b^2 = (2^k * b)^2. This is the square of an even number.
    # We count even m >= 1 such that m^2 <= N, which means m <= sqrt(N).
    # The number of even integers up to limit is limit // 2.
    count_a_even = math.isqrt(N) // 2

    # The two sets are disjoint, so the total count is their sum.
    total_count = count_a_odd + count_a_even
    
    # Print the final result.
    print(total_count)

solve()