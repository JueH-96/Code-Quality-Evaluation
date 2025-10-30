# YOUR CODE HERE
def floyd_warshall(n, dist):
    # Create a copy to avoid modifying the original
    d = [[float('inf')] * (n + 1) for _ in range(n + 1)]
    
    # Initialize with direct edges
    for i in range(1, n + 1):
        d[i][i] = 0
        for j in range(1, n + 1):
            if dist[i][j] != float('inf'):
                d[i][j] = dist[i][j]
    
    # Floyd-Warshall algorithm
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if d[i][k] != float('inf') and d[k][j] != float('inf'):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    
    return d

# Read input
n, m, q = map(int, input().split())

# Store road information
roads = [None]  # 1-indexed
for i in range(m):
    a, b, c = map(int, input().split())
    roads.append((a, b, c))

# Track which roads are closed
closed = [False] * (m + 1)

# Function to build adjacency matrix based on current state
def build_adjacency():
    adj = [[float('inf')] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        adj[i][i] = 0
    
    for i in range(1, m + 1):
        if not closed[i]:
            a, b, c = roads[i]
            adj[a][b] = min(adj[a][b], c)
            adj[b][a] = min(adj[b][a], c)
    
    return adj

# Compute initial shortest paths
adj = build_adjacency()
shortest = floyd_warshall(n, adj)

# Process queries
for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Close road
        road_id = query[1]
        closed[road_id] = True
        
        # Rebuild adjacency and recompute shortest paths
        adj = build_adjacency()
        shortest = floyd_warshall(n, adj)
    
    else:
        # Query shortest path
        x, y = query[1], query[2]
        if shortest[x][y] == float('inf'):
            print(-1)
        else:
            print(shortest[x][y])