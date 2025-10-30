def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    # Initialize adjacency matrix
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    
    # Read edges
    for i in range(M):
        A = int(data[2*i + 2]) - 1
        B = int(data[2*i + 3]) - 1
        matrix[A][B] = 1
    
    # Floyd-Warshall algorithm for transitive closure
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if matrix[i][k] and matrix[k][j]:
                    matrix[i][j] = 1
    
    # Find nodes with no incoming edges in the transitive closure
    candidates = []
    for B in range(N):
        if all(matrix[A][B] == 0 for A in range(N) if A != B):
            candidates.append(B + 1)
    
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()