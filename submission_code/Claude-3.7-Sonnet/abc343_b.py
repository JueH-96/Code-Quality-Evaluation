def main():
    # Read the number of vertices
    N = int(input().strip())
    
    # Read the adjacency matrix
    adjacency_matrix = []
    for _ in range(N):
        row = list(map(int, input().strip().split()))
        adjacency_matrix.append(row)
    
    # For each vertex, find and print its connected vertices
    for i in range(N):
        connected_vertices = []
        for j in range(N):
            if adjacency_matrix[i][j] == 1:
                connected_vertices.append(j + 1)  # Vertices are labeled from 1 to N
        
        # Print the connected vertices (already in ascending order)
        print(' '.join(map(str, connected_vertices)))

if __name__ == "__main__":
    main()