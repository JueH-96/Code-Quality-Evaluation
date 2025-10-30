from heapq import heappush, heappop
import sys
from collections import defaultdict

def dijkstra(graph, start, n):
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
            
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(pq, (dist[v], v))
    
    return dist

def solve():
    # Read input
    N, M = map(int, input().split())
    edges = []
    for _ in range(M):
        a, b, c = map(int, input().split())
        edges.append((a, b, c))
    
    # Find shortest path with all roads
    graph = defaultdict(list)
    for a, b, c in edges:
        graph[a].append((b, c))
        graph[b].append((a, c))
    
    # Get original shortest distance
    original_dist = dijkstra(graph, 1, N)[N]
    
    # For each edge, remove it and find shortest path
    for i in range(M):
        # Create new graph without edge i
        new_graph = defaultdict(list)
        for j, (a, b, c) in enumerate(edges):
            if i != j:
                new_graph[a].append((b, c))
                new_graph[b].append((a, c))
        
        # Find new shortest distance
        new_dist = dijkstra(new_graph, 1, N)[N]
        
        # Compare distances
        if new_dist == original_dist:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    solve()