# YOUR CODE HERE
N = int(input())
adjacency_matrix = []

# Read the adjacency matrix
for i in range(N):
    row = list(map(int, input().split()))
    adjacency_matrix.append(row)

# For each vertex, find connected vertices
for i in range(N):
    connected = []
    for j in range(N):
        if adjacency_matrix[i][j] == 1:
            connected.append(j + 1)  # Convert to 1-indexed
    
    # Print connected vertices (already in ascending order since we iterate j from 0 to N-1)
    if connected:
        print(' '.join(map(str, connected)))
    else:
        print()  # Empty line if no connections