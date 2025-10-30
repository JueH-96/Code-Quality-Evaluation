# YOUR CODE HERE
import sys

# Read input A and B from standard input
# The input consists of two integers A and B separated by a space on a single line.
# map(int, ...) applies the int function to each element returned by split().
a, b = map(int, sys.stdin.readline().split())

# Use a set to store the possible unique values of x.
# A set automatically handles duplicates, so we only store distinct values.
possible_x = set()

# An arithmetic sequence p, q, r satisfies q - p = r - q, or 2q = p + r.
# We are looking for integers x such that A, B, and x can form an arithmetic sequence.
# This means one of A, B, or x must be the middle term (q).

# Case 1: B is the middle term in the arithmetic sequence.
# The sequence could be arranged as (A, B, x) or (x, B, A).
# In both arrangements, the arithmetic property requires 2 * B = A + x.
# Solving for x gives x = 2 * B - A.
x1 = 2 * b - a
possible_x.add(x1)

# Case 2: A is the middle term in the arithmetic sequence.
# The sequence could be arranged as (B, A, x) or (x, A, B).
# In both arrangements, the arithmetic property requires 2 * A = B + x.
# Solving for x gives x = 2 * A - B.
x2 = 2 * a - b
possible_x.add(x2)

# Case 3: x is the middle term in the arithmetic sequence.
# The sequence could be arranged as (A, x, B) or (B, x, A).
# In both arrangements, the arithmetic property requires 2 * x = A + B.
# Solving for x gives x = (A + B) / 2.
# For x to be an integer, the sum A + B must be an even number.
# This occurs if and only if A and B have the same parity (both even or both odd).
# We check this using the modulo operator: (A + B) % 2 == 0.
if (a + b) % 2 == 0:
    # If A + B is even, calculate x using integer division // to ensure the result is an integer.
    x3 = (a + b) // 2
    # Add this possible value of x to the set.
    possible_x.add(x3)

# The final answer is the number of distinct integers x found.
# This is simply the size of the set `possible_x`.
print(len(possible_x))