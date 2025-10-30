# YOUR CODE HERE
A, B = map(int, input().split())

# The 3x3 board is conceptually laid out as:
# 1 2 3
# 4 5 6
# 7 8 9

# For two squares with numbers A and B (where A < B) to be adjacent horizontally,
# two conditions must be met:

# Condition 1: B must be immediately to the right of A.
# This means B must be A + 1.
is_consecutive = (B == A + 1)

# Condition 2: A and B must be in the same row.
# The rows end with 3, 6, and 9.
# If A is 3, its horizontal neighbor B=4 is in the next row.
# If A is 6, its horizontal neighbor B=7 is in the next row.
# (A cannot be 9 because B must be greater than A and less than or equal to 9).
# So, for A and B to be in the same row (given B = A + 1), A must not be
# the last element of a row (i.e., A cannot be 3 or 6).
# This can be checked by verifying that A is not a multiple of 3.
# (For 1-indexed numbers, if A % 3 == 0, then A is 3, 6, or 9).
is_same_row_and_can_be_adjacent = (A % 3 != 0)

# If both conditions are true, then A and B are adjacent horizontally.
if is_consecutive and is_same_row_and_can_be_adjacent:
    print("Yes")
else:
    print("No")