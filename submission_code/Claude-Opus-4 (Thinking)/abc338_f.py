def solve():
    N, M = map(int, input().split())
    
    # Initialize adjacency list
    adj = [[] for _ in range(N)]
    
    # Read edges
    for _ in range(M):
        u, v, w = map(int, input().split())
        u -= 1  # Convert to 0-indexed
        v -= 1
        adj[u].append((v, w))
    
    # DP with bitmask
    # dp[mask][v] = minimum cost to visit all vertices in mask and end at vertex v
    INF = float('inf')
    dp = [[INF] * N for _ in range(1 << N)]
    
    # Base case: start from each vertex
    for v in range(N):
        dp[1 << v][v] = 0
    
    # Fill DP table
    for mask in range(1 << N):
        for v in range(N):
            if dp[mask][v] == INF:
                continue
            # Try to move to each neighbor
            for u, w in adj[v]:
                new_mask = mask | (1 << u)
                dp[new_mask][u] = min(dp[new_mask][u], dp[mask][v] + w)
    
    # Find the minimum cost to visit all vertices
    full_mask = (1 << N) - 1
    min_cost = min(dp[full_mask])
    
    if min_cost == INF:
        print("No")
    else:
        print(min_cost)

solve()