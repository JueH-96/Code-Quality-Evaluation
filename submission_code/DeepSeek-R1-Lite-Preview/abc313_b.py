def main():
    import sys
    sys.setrecursionlimit(1000000)
    N, M = map(int, sys.stdin.readline().split())
    
    # Initialize adjacency matrix
    matrix = [[False] * (N + 1) for _ in range(N + 1)]
    
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        matrix[A][B] = True
    
    # Floyd-Warshall to compute transitive closure
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if matrix[i][k] and matrix[k][j]:
                    matrix[i][j] = True
    
    # Find nodes with no incoming edges in the closure
    candidates = []
    for j in range(1, N + 1):
        has_incoming = False
        for i in range(1, N + 1):
            if i != j and matrix[i][j]:
                has_incoming = True
                break
        if not has_incoming:
            candidates.append(j)
    
    # Determine the output
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()