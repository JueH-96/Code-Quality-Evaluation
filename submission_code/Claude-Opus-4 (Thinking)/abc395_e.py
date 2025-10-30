import heapq
from collections import defaultdict

N, M, X = map(int, input().split())

# Build adjacency lists for both original and reversed graphs
adj = [defaultdict(list), defaultdict(list)]

for _ in range(M):
    u, v = map(int, input().split())
    adj[0][u].append(v)  # Original graph
    adj[1][v].append(u)  # Reversed graph

# Dijkstra's algorithm
pq = [(0, 1, 0)]  # (cost, vertex, layer)
dist = defaultdict(lambda: float('inf'))
dist[(1, 0)] = 0

while pq:
    cost, v, layer = heapq.heappop(pq)
    
    if v == N:
        print(cost)
        break
    
    if cost > dist[(v, layer)]:
        continue
    
    # Move along edges in current layer
    for u in adj[layer][v]:
        if cost + 1 < dist[(u, layer)]:
            dist[(u, layer)] = cost + 1
            heapq.heappush(pq, (cost + 1, u, layer))
    
    # Switch layers (reverse all edges)
    if cost + X < dist[(v, 1 - layer)]:
        dist[(v, 1 - layer)] = cost + X
        heapq.heappush(pq, (cost + X, v, 1 - layer))