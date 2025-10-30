import sys
from collections import defaultdict

def floyd_warshall(n, edges):
    # Initialize distance matrix
    INF = float('inf')
    dist = [[INF] * (n + 1) for _ in range(n + 1)]
    
    # Distance from a city to itself is 0
    for i in range(1, n + 1):
        dist[i][i] = 0
    
    # Add edges
    for a, b, c in edges:
        dist[a][b] = min(dist[a][b], c)
        dist[b][a] = min(dist[b][a], c)
    
    # Floyd-Warshall algorithm
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

# Read input
n, m, q = map(int, input().split())

roads = []
for i in range(m):
    a, b, c = map(int, input().split())
    roads.append((a, b, c))

closed_roads = set()

# Process queries
for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Close road i (1-indexed)
        road_idx = query[1] - 1
        closed_roads.add(road_idx)
    
    else:
        # Find shortest distance from x to y
        x, y = query[1], query[2]
        
        # Build list of open roads
        open_roads = []
        for i, (a, b, c) in enumerate(roads):
            if i not in closed_roads:
                open_roads.append((a, b, c))
        
        # Run Floyd-Warshall on open roads
        dist = floyd_warshall(n, open_roads)
        
        # Output result
        if dist[x][y] == float('inf'):
            print(-1)
        else:
            print(dist[x][y])