def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    photos = data[2:]
    
    # Adjacency matrix - whether two people have ever stood next to each other
    # We will use 0-based indexing internally
    adjacent = [[False]*N for _ in range(N)]
    
    idx = 0
    for _ in range(M):
        lineup = list(map(lambda x: int(x)-1, photos[idx:idx+N]))
        idx += N
        for i in range(N-1):
            p1, p2 = lineup[i], lineup[i+1]
            adjacent[p1][p2] = True
            adjacent[p2][p1] = True
    
    # Count pairs (i,j) with i < j that are never next to each other
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            if not adjacent[i][j]:
                count += 1
    
    print(count)

# Remember to call main()
if __name__ == "__main__":
    main()