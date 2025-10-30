def main():
    import sys
    data = sys.stdin.read().split()
    N, M = map(int, data[:2])
    arr = list(map(int, data[2:]))
    
    # adj[x][y] will be True if x and y have been adjacent in at least one photo
    adj = [[False]*(N+1) for _ in range(N+1)]
    
    idx = 0
    for _ in range(M):
        # read one photo
        photo = arr[idx:idx+N]
        idx += N
        # mark adjacency in this photo
        for j in range(N-1):
            u = photo[j]
            v = photo[j+1]
            adj[u][v] = True
            adj[v][u] = True
    
    # count pairs (x, y), x < y, that never stood adjacent
    count = 0
    for x in range(1, N+1):
        for y in range(x+1, N+1):
            if not adj[x][y]:
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()