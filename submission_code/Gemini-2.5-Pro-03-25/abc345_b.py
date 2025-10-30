# YOUR CODE HERE
import sys

# Read the integer X from standard input.
# The input X can be very large (up to 10^18), so we need to use a data type
# that can handle such large numbers. Python's built-in int type handles
# arbitrary precision integers, so it's suitable.
X = int(sys.stdin.readline())

# We need to compute the ceiling of X / 10, denoted as ceil(X / 10).
# The ceiling function ceil(a) gives the smallest integer greater than or equal to a.

# We can compute the ceiling using integer arithmetic to avoid potential
# floating-point precision issues with very large numbers.

# Method 1: Using the formula ceil(a / b) = (a + b - 1) // b for positive integer b.
# In Python, // performs floor division.
# Here, a = X and b = 10. So, ceil(X / 10) = (X + 10 - 1) // 10 = (X + 9) // 10.
# Let's verify this formula works for both positive and negative X in Python.
# Python's integer division // satisfies a = b * (a // b) + (a % b), where 0 <= a % b < b if b > 0.
# Let X = 10 * q + r, where q = X // 10 and r = X % 10, with 0 <= r < 10.
# We want to compute ceil(X / 10) = ceil((10q + r) / 10) = ceil(q + r / 10).
# Case 1: r = 0 (X is a multiple of 10).
#   ceil(q + 0 / 10) = ceil(q) = q.
#   The formula gives (10q + 9) // 10 = floor((10q + 9) / 10) = floor(q + 0.9) = q. Matches.
# Case 2: r > 0 (1 <= r <= 9).
#   ceil(q + r / 10). Since 0 < r / 10 < 1, q < q + r / 10 < q + 1.
#   The smallest integer >= q + r / 10 is q + 1.
#   The formula gives (10q + r + 9) // 10 = floor((10q + r + 9) / 10) = floor(q + (r + 9) / 10).
#   Since 1 <= r <= 9, we have 10 <= r + 9 <= 18. Thus 1 <= (r + 9) / 10 <= 1.8.
#   floor((r + 9) / 10) = 1.
#   The formula result is q + 1. Matches.
# This formula works correctly for all integer X.

result = (X + 9) // 10

# Method 2 (Alternative): Using the identity ceil(a) = -floor(-a).
# ceil(X / 10) = -floor(-X / 10).
# In Python integer division: ceil(X / 10) = -(-X // 10).
# This method also works correctly.
# result_alt = -(-X // 10)
# assert result == result_alt # This assertion would hold true.

# Print the calculated ceiling value to standard output.
print(result)