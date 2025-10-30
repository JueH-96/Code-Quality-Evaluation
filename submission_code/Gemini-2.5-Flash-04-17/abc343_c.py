import sys

def is_palindrome(n):
    """Checks if a number is a palindrome."""
    s = str(n)
    return s == s[::-1]

# Read the input integer N
# Use strip() to remove potential trailing newline character from readline
N = int(sys.stdin.readline().strip())

# Initialize the maximum palindromic cube found so far.
# Since N is positive, at least 1^3 = 1 is a candidate if N >= 1.
# Initializing to 0 is safe as 1 > 0 and will be updated if N >= 1.
max_palindromic_cube = 0

# We are looking for K = x^3 such that K <= N.
# This means x^3 <= N, so x <= N^(1/3).
# N <= 10^18, so N^(1/3) <= (10^18)^(1/3) = 10^6.
# The maximum value of x we need to check is around 10^6.
# We can estimate an upper bound for x using N^(1/3).
# We add a small margin (e.g., +2) to the integer part of N^(1/3)
# to account for potential floating-point inaccuracies when N is
# close to a perfect cube boundary (M^3).
# For example, if N = M^3 - 1, N^(1/3) is slightly less than M.
# int(N**(1/3)) could be M-1. We need to check up to x=M-1.
# If N = M^3, N^(1/3) is M. int(N**(1/3)) is M. We need to check up to x=M.
# Adding 2 to int(N**(1/3)) ensures the loop range covers the necessary
# integer cube root even with small float errors.
# x must be a positive integer, so we start iterating from x = 1.
x_upper_bound_estimate = int(N**(1.0/3.0)) + 2

# Iterate through possible positive integer values of x.
# The range is inclusive of 1, exclusive of x_upper_bound_estimate.
# We iterate upwards from 1 as it naturally allows finding the maximum
# by updating the max_palindromic_cube value.
for x in range(1, x_upper_bound_estimate):
    # Calculate the cube K = x^3
    k = x**3

    # If the cube exceeds N, any larger x will also result in a cube > N.
    # Since x^3 is strictly increasing for positive x, we can break the loop early.
    if k > N:
        break

    # Check if K is a palindrome.
    if is_palindrome(k):
        # If it's a palindromic cube and K <= N (already checked by the break condition),
        # update the maximum found so far.
        max_palindromic_cube = max(max_palindromic_cube, k)

# Print the result, which is the maximum palindromic cube number not greater than N.
print(max_palindromic_cube)