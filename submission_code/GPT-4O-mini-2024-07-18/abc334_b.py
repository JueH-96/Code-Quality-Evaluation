def count_christmas_trees(A, M, L, R):
    # Find the first tree position >= L
    if L <= A:
        first_tree = A
    else:
        # Calculate the first tree position >= L
        first_tree = A + ((L - A + M - 1) // M) * M
    
    # Find the last tree position <= R
    if R < A:
        last_tree = A
    else:
        # Calculate the last tree position <= R
        last_tree = A + (R - A) // M * M
    
    # If the first tree is greater than the last tree, there are no trees in the range
    if first_tree > last_tree:
        return 0
    
    # Calculate the number of trees
    return (last_tree - first_tree) // M + 1

# Read input
import sys
input = sys.stdin.read
A, M, L, R = map(int, input().strip().split())

# Get the result and print it
result = count_christmas_trees(A, M, L, R)
print(result)