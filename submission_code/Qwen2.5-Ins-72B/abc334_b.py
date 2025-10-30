# YOUR CODE HERE
A, M, L, R = map(int, input().split())

# Calculate the first position of the Christmas tree within the range [L, R]
first_tree = (L - A + M - 1) // M * M + A

# Calculate the last position of the Christmas tree within the range [L, R]
last_tree = (R - A) // M * M + A

# If the first tree is beyond the range, there are no trees
if first_tree > R:
    print(0)
else:
    # Calculate the number of trees
    num_trees = (last_tree - first_tree) // M + 1
    print(num_trees)