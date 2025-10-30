import math
import sys

# Function to calculate floor(sqrt(x)) for non-negative integer x.
# Handles large integers up to 10^18 safely.
def integer_sqrt(x):
    """Calculates floor(sqrt(x)) for non-negative integer x."""
    if x < 0:
        raise ValueError("Input must be non-negative")
    if x == 0:
        return 0

    # Use math.isqrt if available (Python 3.8+)
    if hasattr(math, 'isqrt'):
         return math.isqrt(x)

    # Fallback for older Python versions: Use float approximation and adjust
    # For x <= 10^18, math.sqrt(x) result is <= 10^9.
    # The initial guess s_approx is close to the true floor(sqrt(x)).
    # This approach is reliable for numbers up to the square of the limit
    # of float precision, which is larger than 10^18.
    s_approx = int(math.sqrt(x))
    
    # The true floor(sqrt(x)) is typically s_approx, s_approx-1, or s_approx+1.
    # Check if s_approx + 1 is the correct floor.
    # Check (s_approx + 1)^2 <= x using integer arithmetic.
    # For x <= 10^18, s_approx <= 10^9, (s_approx+1)^2 approx 10^18, fits in Python int.
    # Using multiplication `(s_approx + 1) * (s_approx + 1)` avoids float issues.
    if (s_approx + 1) * (s_approx + 1) <= x:
        return s_approx + 1
    
    # Check if s_approx is the correct floor.
    # Check s_approx^2 > x using integer arithmetic.
    # If s_approx^2 > x, then s_approx is too large, the correct floor is s_approx - 1.
    # Otherwise (s_approx^2 <= x), s_approx is the correct floor.
    if s_approx * s_approx > x:
        return s_approx - 1
    else:
        return s_approx


# Read input from stdin
# Using sys.stdin.readline() is generally faster for competitive programming inputs.
N = int(sys.stdin.readline())

count = 0

# A positive integer X is good if X = 2^a * b^2 for positive integers a, b.
# This is equivalent to X = 2^alpha * k^2, where alpha >= 1 and k is an odd integer >= 1.
# We count the number of integers X such that 1 <= X <= N and X has this form.

# We can iterate through possible values of 2^alpha for alpha >= 1.
# Let P = 2^alpha. We need P * k^2 <= N for some odd k >= 1.
# The smallest possible value for k^2 is 1^2 = 1.
# So we need P * 1 <= N, which means P <= N.
# We iterate P through powers of 2: 2^1, 2^2, 2^3, ... as long as P <= N.

pow2 = 2 # Start with 2^1 (corresponding to alpha = 1)

# Use a while loop that continues as long as the current power of 2 is relevant (i.e., pow2 <= N).
while True:
    # If the current power of 2 exceeds N, no number of the form pow2 * k^2 (k>=1) can be <= N.
    # We can stop iterating through higher powers of 2.
    if pow2 > N:
        break

    # We need P * k^2 <= N, which is k^2 <= N / P.
    # The upper limit for k^2 is floor(N / P).
    limit_sq = N // pow2

    # We need to find the number of odd integers k such that 1 <= k and k^2 <= limit_sq.
    # This is equivalent to finding the number of odd integers k such that 1 <= k <= floor(sqrt(limit_sq)).
    # Let S = floor(sqrt(limit_sq)).
    S = integer_sqrt(limit_sq)

    # We need to count odd integers k such that 1 <= k <= S.
    # The sequence of odd integers is 1, 3, 5, ...
    # The count of odd integers from 1 up to S is given by (S + 1) // 2 using integer division.
    # Examples:
    # S=0: k in [1,0] -> empty set, count = 0. (0+1)//2 = 0. Correct.
    # S=1: k in [1,1] -> {1}, count = 1. (1+1)//2 = 1. Correct.
    # S=2: k in [1,2] -> {1}, count = 1. (2+1)//2 = 1. Correct.
    # S=3: k in [1,3] -> {1,3}, count = 2. (3+1)//2 = 2. Correct.
    count += (S + 1) // 2

    # Prepare for the next iteration: move to the next power of 2.
    # The next power of 2 is current_pow2 * 2.
    # Python's arbitrary precision integers handle potentially large values of pow2.
    # The loop condition `if pow2 > N:` at the beginning will correctly terminate
    # the loop when the updated pow2 is too large.
    
    # Calculate the next power of 2.
    pow2_next = pow2 * 2

    # Update pow2 for the next iteration.
    pow2 = pow2_next

# Print the total count of good integers.
print(count)