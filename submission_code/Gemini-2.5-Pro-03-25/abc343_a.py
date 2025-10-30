# YOUR CODE HERE
import sys

# Read the two integers A and B from standard input
a, b = map(int, sys.stdin.readline().split())

# Calculate the sum S = A + B
s = a + b

# We need to print any integer x such that 0 <= x <= 9 and x != s.
# We know from the constraints that 0 <= s <= 9.
# Consider the integer 0.
# If s is not 0 (i.e., 1 <= s <= 9), then 0 is a valid output because 0 <= 0 <= 9 and 0 != s.
# If s is 0, then 0 is not a valid output. In this case, consider the integer 1.
# Since s = 0, 1 is a valid output because 0 <= 1 <= 9 and 1 != s (1 != 0).

# Therefore, we can use the following logic:
if s == 0:
    # If the sum is 0, print 1 (since 1 is between 0 and 9 and 1 != 0)
    print(1)
else:
    # If the sum is not 0 (it must be between 1 and 9), print 0
    # (since 0 is between 0 and 9 and 0 != s)
    print(0)