from collections import deque
from itertools import combinations

def shortest_path_01_bfs(graph, start, end, n):
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    dq = deque([start])
    
    while dq:
        u = dq.popleft()
        
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    
    return dist[end]

N, M, K = map(int, input().split())
edges = []
for _ in range(M):
    u, v = map(int, input().split())
    edges.append((u, v))

max_dist = 0

for selected_edges in combinations(range(M), K):
    graph = [[] for _ in range(N + 1)]
    
    for i, (u, v) in enumerate(edges):
        if i in selected_edges:
            graph[u].append((v, 1))
        else:
            graph[u].append((v, 0))
    
    dist = shortest_path_01_bfs(graph, 1, N, N)
    max_dist = max(max_dist, dist)

print(max_dist)