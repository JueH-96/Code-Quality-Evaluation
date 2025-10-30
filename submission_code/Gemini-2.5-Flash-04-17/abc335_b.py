# YOUR CODE HERE
import sys

# Read the input N
N = int(sys.stdin.readline())

# Iterate through all possible non-negative values for x, y, and z
# such that x + y + z <= N.
# The nested loop structure ensures lexicographical order.

# x can range from 0 to N.
for x in range(N + 1):
    # For a fixed x, y can range from 0 to N - x.
    for y in range(N - x + 1):
        # For fixed x and y, z can range from 0 to N - x - y.
        for z in range(N - x - y + 1):
            # Print the triple (x, y, z)
            print(x, y, z)