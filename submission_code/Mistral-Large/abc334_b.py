import sys

def count_christmas_trees(A, M, L, R):
    # Calculate the first tree position greater than or equal to L
    first_tree = ((L - A + M - 1) // M) * M + A
    # Calculate the last tree position less than or equal to R
    last_tree = (R - A) // M * M + A

    # Count the number of trees between first_tree and last_tree inclusive
    if first_tree > R or last_tree < L:
        return 0
    else:
        return (last_tree - first_tree) // M + 1

# Read input from stdin
input = sys.stdin.read()
A, M, L, R = map(int, input.split())

# Calculate and print the result
result = count_christmas_trees(A, M, L, R)
print(result)