# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    
    # Initialize adjacency matrix with infinity
    INF = float('inf')
    dist = [[INF] * N for _ in range(N)]
    
    # Direct edges
    direct = [[INF] * N for _ in range(N)]
    
    for i in range(N):
        dist[i][i] = 0
    
    # Read edges
    for _ in range(M):
        u, v, w = map(int, input().split())
        u -= 1  # Convert to 0-indexed
        v -= 1
        direct[u][v] = min(direct[u][v], w)
        dist[u][v] = min(dist[u][v], w)
    
    # Floyd-Warshall to find shortest paths
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Check if graph is strongly connected (all vertices reachable from all vertices)
    for i in range(N):
        for j in range(N):
            if dist[i][j] == INF:
                print("No")
                return
    
    # DP with bitmask
    # dp[mask][v] = minimum cost to visit all vertices in mask, ending at v
    dp = [[INF] * N for _ in range(1 << N)]
    
    # Base case: start from any vertex
    for v in range(N):
        dp[1 << v][v] = 0
    
    # Fill DP table
    for mask in range(1, 1 << N):
        for v in range(N):
            if not (mask & (1 << v)):
                continue
            if dp[mask][v] == INF:
                continue
            
            # Try to extend to vertex u
            for u in range(N):
                if mask & (1 << u):
                    continue
                
                # Use shortest path from v to u
                if dist[v][u] != INF:
                    new_mask = mask | (1 << u)
                    dp[new_mask][u] = min(dp[new_mask][u], dp[mask][v] + dist[v][u])
    
    # Find minimum cost among all ending vertices
    full_mask = (1 << N) - 1
    result = min(dp[full_mask][v] for v in range(N))
    
    if result == INF:
        print("No")
    else:
        print(result)

solve()