def floyd_warshall(n, edges, closed_roads):
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Distance from a city to itself is 0
    for i in range(n):
        dist[i][i] = 0
    
    # Add edges that are not closed
    for i, (a, b, c) in enumerate(edges):
        if i not in closed_roads:
            a -= 1  # Convert to 0-indexed
            b -= 1
            dist[a][b] = min(dist[a][b], c)
            dist[b][a] = min(dist[b][a], c)
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

# Read input
n, m, q = map(int, input().split())

edges = []
for _ in range(m):
    a, b, c = map(int, input().split())
    edges.append((a, b, c))

closed_roads = set()
dist = floyd_warshall(n, edges, closed_roads)

for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Close road i
        i = query[1] - 1  # Convert to 0-indexed
        closed_roads.add(i)
        # Recompute shortest paths
        dist = floyd_warshall(n, edges, closed_roads)
    else:
        # Query shortest distance
        x, y = query[1] - 1, query[2] - 1  # Convert to 0-indexed
        if dist[x][y] == float('inf'):
            print(-1)
        else:
            print(dist[x][y])