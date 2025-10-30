# YOUR CODE HERE
import sys

# Read the three integers from standard input
# sys.stdin.readline() reads a line from stdin
# split() splits the line into a list of strings by whitespace
# map(int, ...) converts each string in the list to an integer
a, b, c = map(int, sys.stdin.readline().split())

# We need to check if any permutation (B_1, B_2, B_3) of (a, b, c) satisfies B_1 * B_2 = B_3.
# The possible ways to assign a, b, and c to (B_1, B_2, B_3) are permutations.
# There are 3! = 6 permutations.
# However, due to the commutative property of multiplication (B_1 * B_2 = B_2 * B_1),
# the order of B_1 and B_2 doesn't matter for the product.
# This means we only need to consider the unique pairs that can form B_1 and B_2.
# The possible pairs from the set {a, b, c} are {a, b}, {a, c}, and {b, c}.
# For each pair {X, Y}, the remaining number Z must be B_3.
# We check if X * Y == Z for each such combination.

# The three distinct checks required are:
# 1. Is a * b equal to c? (Corresponds to permutations like (a, b, c) or (b, a, c))
# 2. Is a * c equal to b? (Corresponds to permutations like (a, c, b) or (c, a, b))
# 3. Is b * c equal to a? (Corresponds to permutations like (b, c, a) or (c, b, a))

# If any of these conditions are true, then it is possible.
if (a * b == c) or (a * c == b) or (b * c == a):
    # If any of the conditions evaluate to True, it means we found a permutation B
    # where the product of the first two elements equals the third.
    print("Yes")
else:
    # If none of the conditions are true after checking all valid pairings,
    # then no such permutation B exists that satisfies the equation.
    print("No")