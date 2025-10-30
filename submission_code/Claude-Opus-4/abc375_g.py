import heapq
from collections import defaultdict

def dijkstra(graph, start, end, n):
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return dist[end]

# Read input
n, m = map(int, input().split())
edges = []
for i in range(m):
    a, b, c = map(int, input().split())
    edges.append((a, b, c))

# Build graph with all edges
graph_all = defaultdict(list)
for a, b, c in edges:
    graph_all[a].append((b, c))
    graph_all[b].append((a, c))

# Find shortest distance with all roads
dist_all = dijkstra(graph_all, 1, n, n)

# Check each edge
for i in range(m):
    # Build graph without edge i
    graph_without = defaultdict(list)
    for j in range(m):
        if i != j:
            a, b, c = edges[j]
            graph_without[a].append((b, c))
            graph_without[b].append((a, c))
    
    # Find shortest distance without edge i
    dist_without = dijkstra(graph_without, 1, n, n)
    
    # Compare distances
    if dist_all != dist_without:
        print("Yes")
    else:
        print("No")