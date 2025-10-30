# YOUR CODE HERE
def count_christmas_trees(A, M, L, R):
    # Find the first tree position greater than or equal to L
    if A < L:
        first_tree = A + ((L - A + M - 1) // M) * M
    else:
        first_tree = A
    
    # Find the last tree position less than or equal to R
    if A > R:
        last_tree = A - ((A - R + M - 1) // M) * M
    else:
        last_tree = A + ((R - A) // M) * M
    
    if first_tree > R or last_tree < L:
        return 0
    
    # Calculate the number of trees between first_tree and last_tree
    return (last_tree - first_tree) // M + 1

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
A = int(data[0])
M = int(data[1])
L = int(data[2])
R = int(data[3])

# Output the result
print(count_christmas_trees(A, M, L, R))