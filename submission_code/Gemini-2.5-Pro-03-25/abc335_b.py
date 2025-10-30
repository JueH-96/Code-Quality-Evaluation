# YOUR CODE HERE
import sys

# Read the integer N from standard input.
# N represents the upper bound for the sum x + y + z.
N = int(sys.stdin.readline())

# We need to find all triples of non-negative integers (x, y, z)
# such that x + y + z <= N.
# The triples must be printed in ascending lexicographical order.

# We can achieve this by using three nested loops.
# The loops iterate through possible values of x, y, and z in order.
# This order of iteration naturally generates the triples in lexicographical order.

# The outermost loop iterates through possible values for x.
# Since x, y, z are non-negative and their sum is at most N,
# x itself cannot be greater than N.
# So, x ranges from 0 to N (inclusive).
for x in range(N + 1):
    # The middle loop iterates through possible values for y, given the current value of x.
    # Since y >= 0 and x + y + z <= N (with z >= 0), we must have x + y <= N.
    # This means y can range from 0 up to N - x (inclusive).
    for y in range(N - x + 1):
        # The innermost loop iterates through possible values for z, given the current values of x and y.
        # Since z >= 0 and x + y + z <= N, we must have z <= N - x - y.
        # This means z can range from 0 up to N - x - y (inclusive).
        for z in range(N - x - y + 1):
            # At this point, the triple (x, y, z) satisfies all conditions:
            # 1. x, y, z are non-negative integers (because range starts from 0).
            # 2. x + y + z <= N (because z is bounded by N - x - y).
            # 3. The triples are generated in lexicographical order due to the loop structure.
            
            # Print the triple (x, y, z).
            # The print function with multiple arguments automatically separates them with spaces
            # and adds a newline character at the end, matching the required output format.
            print(x, y, z)