from collections import defaultdict
import heapq

def dijkstra(graph, start, n):
    dist = [float('inf')] * (n+1)
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

def solve():
    n = int(input())
    
    # Build adjacency list
    graph = defaultdict(list)
    edges = []
    for _ in range(n-1):
        u, v, l = map(int, input().split())
        graph[u].append((v, l))
        graph[v].append((u, l))
        edges.append((u, v, l))
    
    # Calculate distances from vertex 1 to all other vertices
    distances = dijkstra(graph, 1, n)
    
    # For each K from 1 to N
    for k in range(1, n+1):
        # For K=1, answer is 2 * max distance from vertex 1
        if k == 1:
            ans = 2 * max(distances[1:])
        else:
            # For K≥2, we need to visit all vertices that are farthest from vertex 1
            # Sort vertices by distance from vertex 1
            vertices = [(distances[i], i) for i in range(1, n+1)]
            vertices.sort(reverse=True)
            
            # Take K farthest vertices
            farthest = vertices[:k]
            
            # Answer is 2 * sum of distances to these vertices minus the maximum distance
            total = sum(d for d, _ in farthest)
            ans = 2 * total - max(d for d, _ in farthest)
            
        print(ans)

solve()