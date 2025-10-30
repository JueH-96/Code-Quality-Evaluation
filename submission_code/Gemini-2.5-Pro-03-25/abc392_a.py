# YOUR CODE HERE
import sys

# Read the three integers from standard input
# sys.stdin.readline() reads a line from standard input (including the newline character)
# .split() splits the line into a list of strings based on whitespace
# map(int, ...) applies the int() function to each element in the list, converting strings to integers
# a, b, c = ... unpacks the resulting iterator of integers into three variables
a, b, c = map(int, sys.stdin.readline().split())

# The problem asks if there exists a permutation B = (B_1, B_2, B_3) of A = (a, b, c)
# such that B_1 * B_2 = B_3.

# Let's consider the possible permutations and the conditions they imply:
# Permutation (a, b, c): Requires a * b = c
# Permutation (a, c, b): Requires a * c = b
# Permutation (b, a, c): Requires b * a = c (same as a * b = c)
# Permutation (b, c, a): Requires b * c = a
# Permutation (c, a, b): Requires c * a = b (same as a * c = b)
# Permutation (c, b, a): Requires c * b = a (same as b * c = a)

# Therefore, we only need to check if any of the following three conditions are true:
# 1. a * b == c
# 2. a * c == b
# 3. b * c == a

# We use an if-else statement with the 'or' operator to check these conditions.
# If any one of the conditions is true, it means such a permutation B exists.
if a * b == c or a * c == b or b * c == a:
    # If at least one condition is met, print "Yes"
    print("Yes")
else:
    # If none of the conditions are met after checking all possibilities, print "No"
    print("No")