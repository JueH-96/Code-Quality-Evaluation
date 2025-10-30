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
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
                
    return dist

N = int(input())
graph = defaultdict(list)

for _ in range(N-1):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
    graph[b].append((a, c))

ans = float('inf')
for start in range(1, N+1):
    dist = dijkstra(graph, start, N)
    max_dist = max(d for d in dist[1:] if d != float('inf'))
    ans = min(ans, max_dist)

print(ans * 2 - max(max(c for _, _, c in [map(int, input().split()) for _ in range(N-1)]), 0))