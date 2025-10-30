# YOUR CODE HERE

# Read the inputs
A, M, L, R = map(int, input().split())

# Calculate the number of trees to be set up
if A <= L:
    start = L
else:
    start = A + ((L - A) // M + (1 if (L - A) % M != 0 else 0)) * M

end = R + M

num_trees = (end - start) // M + 1

# Print the number of trees
print(num_trees)