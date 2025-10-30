import sys

# Read the two integers A and B from standard input
line = sys.stdin.readline().split()
A = int(line[0])
B = int(line[1])

# The 3x3 board is arranged as:
# 1 2 3
# 4 5 6
# 7 8 9

# Two squares with numbers A and B are horizontally adjacent if:
# 1. They are consecutive numbers (B must be A + 1, since A < B is given)
# 2. They are located in the same row on the board.

# The numbers 1 through 9 are assigned to the board row by row.
# Row 1 contains numbers 1, 2, 3.
# Row 2 contains numbers 4, 5, 6.
# Row 3 contains numbers 7, 8, 9.

# If B is A + 1, they are horizontally adjacent if and only if A is not the last number in its row.
# The numbers that are the last in their respective rows are 3, 6, and 9.
# Since A < B <= 9, the maximum possible value for A is 8.
# Therefore, A can only be 3 or 6 if it's the last number in a row.
# If A is 3, B is 4. These are not horizontally adjacent (3 is end of row 1, 4 is start of row 2).
# If A is 6, B is 7. These are not horizontally adjacent (6 is end of row 2, 7 is start of row 3).
# For any other value of A (where 1 <= A <= 8 and B = A + 1), A and B will be in the same row.
# A is not 3 and A is not 6 if A is not a multiple of 3 (for A in the range [1, 8]).
# The condition for horizontal adjacency between A and B (A < B) is thus:
# B must be A + 1 AND A must not be a multiple of 3.

# Check if B is equal to A + 1 AND A is not divisible by 3
is_adjacent = (B == A + 1) and (A % 3 != 0)

# Print "Yes" if they are horizontally adjacent, and "No" otherwise.
if is_adjacent:
    print("Yes")
else:
    print("No")