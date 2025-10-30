import sys

def solve():
    # Read N, the number of vertices
    N = int(sys.stdin.readline())

    # Read the adjacency matrix
    # The matrix will be stored as a list of lists.
    # adj_matrix[i][j] will represent the edge between vertex i+1 and vertex j+1.
    adj_matrix = []
    for _ in range(N):
        row = list(map(int, sys.stdin.readline().split()))
        adj_matrix.append(row)

    # For each vertex, find and print its direct neighbors
    # Vertices are labeled 1 to N, but matrix indices are 0 to N-1.
    for i in range(N):  # Loop through each vertex i (0-indexed)
        neighbors = []  # List to store neighbors of the current vertex
        for j in range(N):  # Loop through all possible vertices j (0-indexed)
            # If adj_matrix[i][j] is 1, it means there's an edge between vertex i+1 and vertex j+1
            if adj_matrix[i][j] == 1:
                # Add the 1-indexed vertex number (j+1) to the neighbors list
                neighbors.append(j + 1)
        
        # Print the neighbors.
        # The neighbors list is already in ascending order because we iterate j from 0 to N-1.
        # The * operator unpacks the list elements as separate arguments to print(),
        # which by default prints them space-separated.
        # If 'neighbors' is empty, print() will simply print a newline, which is correct.
        print(*neighbors)

# Call the solve function to run the program
solve()