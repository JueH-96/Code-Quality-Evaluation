import heapq
from collections import defaultdict

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Build adjacency list
graph = defaultdict(list)
for _ in range(M):
    u, v, b = map(int, input().split())
    u -= 1  # Convert to 0-indexed
    v -= 1
    graph[u].append((v, b))
    graph[v].append((u, b))

# Dijkstra's algorithm
dist = [float('inf')] * N
dist[0] = A[0]  # Starting at vertex 1 (0-indexed), we count its weight

# Priority queue: (distance, vertex)
pq = [(A[0], 0)]

while pq:
    d, u = heapq.heappop(pq)
    
    if d > dist[u]:
        continue
    
    for v, edge_weight in graph[u]:
        # New distance = current distance + edge weight + destination vertex weight
        new_dist = dist[u] + edge_weight + A[v]
        
        if new_dist < dist[v]:
            dist[v] = new_dist
            heapq.heappush(pq, (new_dist, v))

# Output results for vertices 2 to N
print(' '.join(map(str, dist[1:])))