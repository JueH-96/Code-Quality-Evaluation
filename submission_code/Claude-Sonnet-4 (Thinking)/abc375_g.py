import heapq
from collections import defaultdict

def dijkstra(graph, start, n):
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
    
    return dist

n, m = map(int, input().split())
edges = []

for i in range(m):
    a, b, c = map(int, input().split())
    edges.append((a, b, c))

# Build full graph
full_graph = defaultdict(list)
for a, b, c in edges:
    full_graph[a].append((b, c))
    full_graph[b].append((a, c))

# Compute shortest distances from city 1 and from city n
dist_from_1 = dijkstra(full_graph, 1, n)
dist_from_n = dijkstra(full_graph, n, n)

shortest_distance = dist_from_1[n]

for i, (a, b, c) in enumerate(edges):
    # Check if edge is on some shortest path from 1 to n
    on_path = (dist_from_1[a] + c + dist_from_n[b] == shortest_distance or
               dist_from_1[b] + c + dist_from_n[a] == shortest_distance)
    
    if not on_path:
        print("No")
    else:
        # Remove edge i and recompute shortest distance
        graph_without_i = defaultdict(list)
        for j in range(m):
            if j != i:
                x, y, z = edges[j]
                graph_without_i[x].append((y, z))
                graph_without_i[y].append((x, z))
        
        new_dist = dijkstra(graph_without_i, 1, n)[n]
        
        if shortest_distance != new_dist:
            print("Yes")
        else:
            print("No")