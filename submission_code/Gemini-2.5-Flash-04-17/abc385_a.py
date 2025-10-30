# YOUR CODE HERE
import sys

# Read the three integers from standard input
line = sys.stdin.readline().split()
A = int(line[0])
B = int(line[1])
C = int(line[2])

# Determine if it is possible to divide A, B, C into two or more groups
# so that these groups have equal sums.

# A partition of 3 elements into >= 2 non-empty groups can have k=2 or k=3 groups.

# Case 1: Partition into 2 groups (k=2).
# The only way to partition {A, B, C} into two non-empty groups is {{x, y}, {z}}.
# For the sums to be equal, sum({x, y}) must equal sum({z}).
# This means one number must be equal to the sum of the other two numbers.
# Check if A = B + C, or B = A + C, or C = A + B.
# e.g., if A + B == C, we can partition into {A, B} and {C}. Their sums are A+B and C.
two_group_possible = (A + B == C) or (A + C == B) or (B + C == A)

# Case 2: Partition into 3 groups (k=3).
# The only way to partition {A, B, C} into three non-empty groups is {{A}, {B}, {C}}.
# For the sums to be equal, sum({A}) must equal sum({B}) must equal sum({C}).
# This means A must equal B, and B must equal C.
# Check if A == B and B == C.
three_group_possible = (A == B and B == C)

# It is possible to divide the integers into two or more groups with equal sums
# if either the two-group possibility or the three-group possibility holds.
if two_group_possible or three_group_possible:
    print("Yes")
else:
    print("No")