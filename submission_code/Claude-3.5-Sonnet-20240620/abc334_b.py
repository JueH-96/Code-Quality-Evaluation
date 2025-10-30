# YOUR CODE HERE
def count_trees(A, M, L, R):
    # Function to find the first tree position >= x
    def first_tree(x):
        return A + M * ((x - A + M - 1) // M)
    
    # Function to find the last tree position <= x
    def last_tree(x):
        return A + M * ((x - A) // M)
    
    # Find the positions of the first and last trees
    first = first_tree(L)
    last = last_tree(R)
    
    # If there are no trees in the range, return 0
    if first > last:
        return 0
    
    # Calculate the number of trees
    return (last - first) // M + 1

# Read input
A, M, L, R = map(int, input().split())

# Calculate and print the result
print(count_trees(A, M, L, R))