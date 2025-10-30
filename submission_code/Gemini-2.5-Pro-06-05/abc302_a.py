import sys

# Read two space-separated integers from standard input.
# sys.stdin.readline is used for efficiency, and map(int, ...) converts the parts to integers.
# Python's int type automatically handles the large numbers specified in the constraints.
A, B = map(int, sys.stdin.readline().split())

# Calculate the minimum number of attacks required.
# This is equivalent to finding the ceiling of A divided by B, i.e., ceil(A/B).
# To avoid floating-point precision errors with large numbers, we use a standard
# integer arithmetic formula for ceiling division: (numerator + denominator - 1) // denominator.
num_attacks = (A + B - 1) // B

# Print the final result to standard output.
print(num_attacks)