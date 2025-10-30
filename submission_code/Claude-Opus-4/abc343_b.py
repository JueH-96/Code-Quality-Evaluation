# YOUR CODE HERE
N = int(input())
adjacency_matrix = []
for i in range(N):
    row = list(map(int, input().split()))
    adjacency_matrix.append(row)

for i in range(N):
    connected_vertices = []
    for j in range(N):
        if adjacency_matrix[i][j] == 1:
            connected_vertices.append(j + 1)  # Convert to 1-indexed
    
    if connected_vertices:
        print(' '.join(map(str, connected_vertices)))
    else:
        print()