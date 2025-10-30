import sys

# Read the integer X from standard input.
# sys.stdin.readline() is used for potentially faster input than input()
# for competitive programming, especially with large inputs, though for a single integer,
# it's not strictly necessary.
X = int(sys.stdin.readline())

# Calculate the ceiling of X/10 using integer arithmetic.
# Python's integer division operator // performs "floor division",
# meaning it truncates towards negative infinity.
# The modulo operator % returns a result with the same sign as the divisor.
# Since our divisor (10) is positive, X % 10 will always be non-negative (0 to 9).

# Case 1: X is a multiple of 10.
# If X is a multiple of 10 (e.g., 40, -20), then X / 10 is an integer.
# In this case, X // 10 correctly gives the ceiling value.
# For example:
#   X = 40: 40 // 10 = 4. ceil(40/10) = 4. (X % 10 == 0)
#   X = -20: -20 // 10 = -2. ceil(-20/10) = -2. (X % 10 == 0)

# Case 2: X is not a multiple of 10.
# If X is not a multiple of 10 (e.g., 27, -13), then X / 10 is not an integer.
# Due to the properties of Python's // and % operators:
# X = (X // 10) * 10 + (X % 10)
# Since X % 10 will be a positive remainder (1 through 9),
# X // 10 will be the integer part rounded down (floor).
# To get the ceiling, we need to add 1 to this floor value.
# For example:
#   X = 27: 27 // 10 = 2. 27 % 10 = 7. ceil(27/10) = 3. So, 2 + 1.
#   X = -13: -13 // 10 = -2. -13 % 10 = 7. ceil(-13/10) = -1. So, -2 + 1.
# In these cases, X % 10 will be non-zero.

if X % 10 == 0:
    result = X // 10
else:
    result = X // 10 + 1

# Print the calculated result to standard output.
print(result)