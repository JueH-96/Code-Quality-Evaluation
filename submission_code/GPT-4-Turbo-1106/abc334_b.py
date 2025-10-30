import math

# Read the inputs from stdin
A, M, L, R = map(int, input().split())

# Calculate the number of Christmas trees between Takahashi and Aoki
# Find the first tree that is within or to the right of L
first_tree = A + math.ceil((L - A) / M) * M if L > A else A + math.floor((L - A) / M) * M

# Calculate the number of trees between the first tree and R
trees_count = (R - first_tree) // M + 1 if first_tree <= R else 0

# Write the answer to stdout
print(trees_count)