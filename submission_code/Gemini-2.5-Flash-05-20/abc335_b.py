# Read the integer N from standard input
N = int(input())

# Iterate through all possible values for x
# x can range from 0 up to N.
# The outer loop ensures that triples are sorted primarily by x.
for x in range(N + 1):
    # For a fixed x, iterate through all possible values for y.
    # y can range from 0 up to N - x (since x + y + z <= N and z >= 0, x + y must be <= N).
    # This loop ensures that for a given x, triples are sorted by y.
    for y in range(N - x + 1):
        # For fixed x and y, iterate through all possible values for z.
        # z can range from 0 up to N - x - y (since x + y + z <= N).
        # This innermost loop ensures that for given x and y, triples are sorted by z.
        for z in range(N - x - y + 1):
            # Print the triple (x, y, z) separated by spaces.
            # Each triple is printed on a new line.
            print(f"{x} {y} {z}")