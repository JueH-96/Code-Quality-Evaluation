# Read the number of vertices
N = int(input())

# Initialize the adjacency matrix
adj_matrix = []

# Read the adjacency matrix
for _ in range(N):
    row = list(map(int, input().split()))
    adj_matrix.append(row)

# For each vertex, find its neighbors
for i in range(N):
    neighbors = []
    for j in range(N):
        if adj_matrix[i][j] == 1:
            neighbors.append(j + 1)  # Since vertices are labeled from 1 to N
    # Print the neighbors in ascending order
    print(' '.join(map(str, sorted(neighbors))))