import heapq
from itertools import combinations

def dijkstra(n, edges, weights):
    # Build adjacency list with weights
    adj = [[] for _ in range(n + 1)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, weights[i]))
    
    # Dijkstra's algorithm
    dist = [float('inf')] * (n + 1)
    dist[1] = 0
    pq = [(0, 1)]  # (distance, vertex)
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return dist[n]

# Read input
n, m, k = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# Try all combinations of k edges to have weight 1
max_dist = 0

for chosen_edges in combinations(range(m), k):
    # Create weight array: 1 for chosen edges, 0 for others
    weights = [0] * m
    for idx in chosen_edges:
        weights[idx] = 1
    
    # Find shortest path with these weights
    dist = dijkstra(n, edges, weights)
    max_dist = max(max_dist, dist)

print(max_dist)