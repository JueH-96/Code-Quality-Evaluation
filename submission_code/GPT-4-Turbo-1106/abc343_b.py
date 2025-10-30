# Read the number of vertices
N = int(input())

# Initialize an empty adjacency matrix
adjacency_matrix = []

# Read the adjacency matrix from standard input
for _ in range(N):
    adjacency_matrix.append(list(map(int, input().split())))

# For each vertex, find the directly connected vertices
for i in range(N):
    # Find the vertices connected to vertex i
    connected_vertices = [str(j + 1) for j in range(N) if adjacency_matrix[i][j] == 1]
    # Print the connected vertices in ascending order
    print(' '.join(connected_vertices))