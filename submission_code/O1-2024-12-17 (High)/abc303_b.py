def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    idx = 2
    
    # Read all photos
    photos = []
    for _ in range(M):
        row = list(map(int, data[idx:idx+N]))
        idx += N
        photos.append(row)
    
    # Create an adjacency matrix to mark pairs that were adjacent in at least one photo
    adj = [[False]*N for _ in range(N)]
    
    # For each photo, mark adjacent pairs
    for i in range(M):
        for j in range(N-1):
            a = photos[i][j] - 1  # convert 1-based to 0-based index
            b = photos[i][j+1] - 1
            adj[a][b] = True
            adj[b][a] = True
    
    # Count pairs that never appeared adjacent in any photo
    ans = 0
    for x in range(N):
        for y in range(x+1, N):
            if not adj[x][y]:
                ans += 1
    
    print(ans)

# Do not forget to call main
if __name__ == "__main__":
    main()