import sys

# Read input
A, M, L, R = map(int, sys.stdin.readline().split())

# We are looking for points x = A + k*M such that L <= x <= R
# This is equivalent to L - A <= k*M <= R - A

# Let y = k*M. We need to count integers y such that y is a multiple of M,
# and L - A <= y <= R - A.

# Let range_start = L - A and range_end = R - A.
# We need to count multiples of M in the inclusive integer range [range_start, range_end].

# The number of multiples of a positive integer m less than or equal to an integer n is floor(n / m).
# In Python integer division, floor(n / m) is n // m.

# Number of multiples of M less than or equal to range_end is range_end // M.

# The number of multiples of M strictly less than range_start is the
# number of multiples of M less than or equal to range_start - 1.
# This is (range_start - 1) // M.

# The number of multiples of M in the range [range_start, range_end] is:
# (Number of multiples <= range_end) - (Number of multiples < range_start)
# = (R - A) // M - (L - A - 1) // M

count = (R - A) // M - (L - A - 1) // M

# The count derived from this formula should naturally be non-negative
# given L <= R and the properties of floor division.
# However, for safety or clarity, one could take max(0, count), although
# with the properties of floor division in Python and L <= R, this seems unnecessary
# as the number of multiples <= N is non-decreasing with N.

# Print the result
print(count)