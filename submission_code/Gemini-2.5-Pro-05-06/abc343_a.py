# Read the two integers A and B from input
A, B = map(int, input().split())

# Calculate their sum
S = A + B

# Determine the integer to print
# We need to print any integer X such that 0 <= X <= 9 and X != S.
# The sum S is guaranteed to be between 0 and 9 (inclusive) due to problem constraints.

# Logic:
# If S is 0, then 1 is a valid choice for X (0 <= 1 <= 9 and 1 != 0).
# If S is not 0 (i.e., S is 1, 2, ..., 9), then 0 is a valid choice for X (0 <= 0 <= 9 and 0 != S).
if S == 0:
    print(1)
else:
    print(0)