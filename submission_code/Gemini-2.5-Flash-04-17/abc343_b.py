import sys

# Read the number of vertices
N = int(sys.stdin.readline())

# Read the adjacency matrix
adj_matrix = []
for _ in range(N):
    # Read a row, split by spaces, convert to integers
    row = list(map(int, sys.stdin.readline().split()))
    adj_matrix.append(row)

# For each vertex (from 1 to N)
for i in range(N):
    # i is the 0-indexed row, corresponding to vertex i+1
    neighbors = []
    # Check each potential neighbor (from 1 to N)
    for j in range(N):
        # j is the 0-indexed column, corresponding to vertex j+1
        # If the entry is 1, there is an edge between vertex i+1 and vertex j+1
        if adj_matrix[i][j] == 1:
            # Add the 1-indexed neighbor vertex number to the list
            neighbors.append(j + 1)

    # The neighbors list is already in ascending order because we iterated j from 0 to N-1
    # Convert the list of integers to a list of strings
    neighbor_strings = map(str, neighbors)
    # Join the strings with a space
    output_line = " ".join(neighbor_strings)
    # Print the result for this vertex
    print(output_line)