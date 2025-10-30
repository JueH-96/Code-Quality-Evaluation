def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    # adj[u][v] will be True if u and v ever stood next to each other
    adj = [[False]*(N+1) for _ in range(N+1)]
    
    for _ in range(M):
        line = list(map(int, input().split()))
        # mark adjacent pairs in this photo
        for i in range(N-1):
            u = line[i]
            v = line[i+1]
            adj[u][v] = True
            adj[v][u] = True
    
    # count pairs (u, v) with u < v that never were adjacent
    count = 0
    for u in range(1, N+1):
        for v in range(u+1, N+1):
            if not adj[u][v]:
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()