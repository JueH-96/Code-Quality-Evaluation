# YOUR CODE HERE
import sys

def solve():
    """
    Reads the graph information (number of vertices and adjacency matrix)
    and prints the neighbors for each vertex in ascending order.
    """
    # Read the number of vertices
    N = int(sys.stdin.readline())

    # Read the adjacency matrix
    # The matrix uses 0-based indexing, while vertices are 1-based (1 to N).
    # adj_matrix[i][j] = 1 means there is an edge between vertex i+1 and vertex j+1.
    adj_matrix = []
    for _ in range(N):
        # Read a row of the adjacency matrix from standard input.
        # Each row contains N integers (0 or 1) separated by spaces.
        # map(int, ...) converts each space-separated string element to an integer.
        # list(...) converts the map object to a list.
        row = list(map(int, sys.stdin.readline().split()))
        adj_matrix.append(row)

    # Process each vertex from 1 to N.
    # The outer loop iterates through the rows of the adjacency matrix.
    # Row index `i` (from 0 to N-1) corresponds to vertex `i+1`.
    for i in range(N):
        neighbors = [] # Initialize an empty list to store the neighbors of vertex i+1.

        # Iterate through the columns of the current row `i`.
        # Column index `j` (from 0 to N-1) corresponds to vertex `j+1`.
        for j in range(N):
            # Check if the entry in the adjacency matrix indicates an edge.
            # adj_matrix[i][j] == 1 signifies an edge between vertex i+1 and vertex j+1.
            if adj_matrix[i][j] == 1:
                # If an edge exists, the vertex `j+1` is a neighbor of vertex `i+1`.
                # Add the neighbor's label (j+1) to the list.
                neighbors.append(j + 1)

        # After checking all columns for row `i`, the `neighbors` list contains
        # all vertices directly connected to vertex `i+1`.
        # Since we iterated through columns `j` from 0 to N-1, the corresponding
        # vertex labels `j+1` are added to the `neighbors` list in ascending order.
        # Therefore, the `neighbors` list is already sorted.

        # Print the neighbors for vertex `i+1`.
        # The `*neighbors` syntax unpacks the list `neighbors` into individual arguments
        # for the `print` function. By default, `print` separates arguments with a space.
        # If `neighbors` is empty (no neighbors), this prints an empty line.
        print(*neighbors)

# This is the standard boilerplate for running the main logic in Python scripts.
# It ensures that the `solve()` function is called only when the script is executed directly.
if __name__ == "__main__":
    solve()