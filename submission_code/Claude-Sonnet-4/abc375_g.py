import heapq
from collections import defaultdict
import sys

def dijkstra(graph, start, end, n):
    # Returns shortest distance from start to end, or float('inf') if unreachable
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            return d
        
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return dist[end]

def solve():
    n, m = map(int, input().split())
    roads = []
    
    # Read all roads
    for i in range(m):
        a, b, c = map(int, input().split())
        roads.append((a, b, c))
    
    # Build complete graph
    full_graph = defaultdict(list)
    for a, b, c in roads:
        full_graph[a].append((b, c))
        full_graph[b].append((a, c))
    
    # Find shortest distance with all roads
    full_distance = dijkstra(full_graph, 1, n, n)
    
    # For each road, check distance without that road
    results = []
    for i in range(m):
        # Build graph without road i
        partial_graph = defaultdict(list)
        for j, (a, b, c) in enumerate(roads):
            if j != i:  # Skip road i
                partial_graph[a].append((b, c))
                partial_graph[b].append((a, c))
        
        # Find shortest distance without road i
        partial_distance = dijkstra(partial_graph, 1, n, n)
        
        # Compare distances
        if full_distance != partial_distance:
            results.append("Yes")
        else:
            results.append("No")
    
    # Output results
    for result in results:
        print(result)

solve()