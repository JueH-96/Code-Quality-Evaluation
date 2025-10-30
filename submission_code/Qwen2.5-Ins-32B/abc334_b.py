import sys

def count_christmas_trees(A, M, L, R):
    # Calculate the number of trees from A to L-1
    start = (L - A - 1) // M
    # Calculate the number of trees from A to R
    end = (R - A) // M
    # The number of trees between L and R is the difference
    return end - start

# Read input from stdin
A, M, L, R = map(int, sys.stdin.readline().split())

# Calculate and print the result
print(count_christmas_trees(A, M, L, R))