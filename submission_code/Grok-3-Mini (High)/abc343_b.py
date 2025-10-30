import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Initialize index to 0
index = 0

# Read N from the first element
N = int(data[index])
index += 1

# Build the adjacency matrix
adj_matrix = []
for i in range(N):
    row = []
    for j in range(N):
        val = int(data[index])
        index += 1
        row.append(val)
    adj_matrix.append(row)

# For each vertex from 1 to N, find and print the neighbors in ascending order
for i in range(1, N + 1):  # Vertex number (1-based)
    # Find neighbors using list comprehension
    neighbors = [j + 1 for j in range(N) if adj_matrix[i - 1][j] == 1]
    # Print neighbors separated by spaces
    print(' '.join(map(str, neighbors)))