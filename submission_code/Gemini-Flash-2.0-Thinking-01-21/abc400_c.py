# YOUR CODE HERE
import sys
import math

# Function to solve the problem
def solve():
    # Read input N as an integer
    # N can be up to 10^18, so use Python's arbitrary-precision integer type
    N = int(sys.stdin.readline())

    # Initialize the count of good integers
    count = 0

    # A positive integer X is good if X = 2^a * b^2 for positive integers a and b.
    # Let the prime factorization of b be b = 2^c * d, where c >= 0 and d is odd.
    # Since b >= 1, d must be >= 1.
    # Then b^2 = 2^(2c) * d^2.
    # X = 2^a * (2^(2c) * d^2) = 2^(a + 2c) * d^2.
    # Let k = a + 2c. Since a >= 1 and c >= 0, k must be a positive integer (k >= 1).
    # Let s = d. Since d is odd and d >= 1, s must be an odd positive integer (s >= 1).
    # So, a good integer X can be uniquely represented as X = 2^k * s^2,
    # where k is a positive integer (k >= 1) and s is an odd positive integer (s >= 1).

    # We need to count the number of such integers X where 1 <= X <= N.
    # This is equivalent to counting pairs (k, s) such that k >= 1, s >= 1 is odd,
    # and 2^k * s^2 <= N.

    # We can iterate through possible values of k (the exponent of 2).
    # Since 2^k * s^2 <= N and s >= 1, we must have 2^k * 1^2 <= N, so 2^k <= N.
    # The smallest value for k is 1 (2^1 = 2).
    # The largest value for k is such that 2^k <= N.
    # We can iterate using the power of 2 directly: 2^1, 2^2, 2^3, ...
    
    # Start with power_of_2 = 2^1 = 2 (corresponding to k=1)
    power_of_2 = 2

    # Loop while the current power of 2 is less than or equal to N.
    # If power_of_2 > N, then 2^k > N, and 2^k * s^2 > N for any s >= 1.
    # So, no good integers <= N can be formed with this k or larger k values.
    while power_of_2 <= N:
        # For the current power_of_2 = 2^k, we need to find the number of odd integers s >= 1
        # such that 2^k * s^2 <= N.
        # This inequality is s^2 <= N / 2^k.

        # Calculate the maximum allowed value for s^2.
        # Use integer division to handle potentially large N and power_of_2.
        M = N // power_of_2

        # We need s >= 1, s is odd, and s^2 <= M.
        # This is equivalent to s >= 1, s is odd, and s <= floor(sqrt(M)).

        # Calculate S = floor(sqrt(M)) using integer square root.
        # math.isqrt(x) computes floor(sqrt(x)) for non-negative integers x.
        # Since power_of_2 >= 2 and N >= 1, M = N // power_of_2 is always >= 0.
        S = math.isqrt(M)

        # We need to count the number of odd integers s in the range [1, S].
        # The number of odd integers s >= 1 such that s <= S is given by (S + 1) // 2.
        # Example: S=0 -> (0+1)//2 = 0 (no odd s in [1,0])
        # Example: S=1 -> (1+1)//2 = 1 (odd s in [1,1] is {1})
        # Example: S=2 -> (2+1)//2 = 1 (odd s in [1,2] is {1})
        # Example: S=3 -> (3+1)//2 = 2 (odd s in [1,3] is {1, 3})
        # This formula is correct for S >= 0.

        num_odd_s = (S + 1) // 2

        # Add the count of valid odd s for the current power of 2 (current k)
        # to the total count of good integers.
        count += num_odd_s

        # Move to the next power of 2 (which corresponds to k+1).
        # Since Python 3 handles arbitrarily large integers, the multiplication
        # `power_of_2 * 2` will not overflow.
        # The loop condition `while power_of_2 <= N` will ensure that
        # we stop when power_of_2 becomes greater than N.
        power_of_2 *= 2

    # Print the final total count of good integers.
    print(count)

# Execute the solve function to run the program
solve()