from heapq import heappush, heappop
import sys

def dijkstra(graph, weights, start, n):
    dist = [float('inf')] * (n+1)
    dist[start] = weights[start]
    
    pq = [(weights[start], start)]
    
    while pq:
        d, u = heappop(pq)
        
        if d > dist[u]:
            continue
            
        for v, w in graph[u]:
            cost = dist[u] + w + weights[v]
            
            if cost < dist[v]:
                dist[v] = cost
                heappush(pq, (cost, v))
                
    return dist

# Read input
n, m = map(int, input().split())
weights = [0] + list(map(int, input().split()))

# Build graph
graph = [[] for _ in range(n+1)]
for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    graph[v].append((u, w))

# Find shortest paths from vertex 1
dist = dijkstra(graph, weights, 1, n)

# Print answers for vertices 2 to n
print(*dist[2:])