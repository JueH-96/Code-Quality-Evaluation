from collections import deque
from math import inf

# Read input
N, M, Q = map(int, input().split())
roads = []
for _ in range(M):
    A, B, C = map(int, input().split())
    roads.append((A, B, C))

# Initialize graph
graph = [[] for _ in range(N+1)]
for a, b, c in roads:
    graph[a].append((b, c))
    graph[b].append((a, c))

# Process queries
closed_roads = [False] * (M+1)
for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        closed_roads[query[1]] = True
    else:
        x, y = query[1], query[2]
        
        # Dijkstra's algorithm
        dist = [inf] * (N+1)
        dist[x] = 0
        queue = deque([(x, 0)])
        while queue:
            u, d = queue.popleft()
            if d > dist[u]:
                continue
            for v, c in graph[u]:
                if not closed_roads[next((i for i, (a, b, _) in enumerate(roads) if (a, b) == (u, v) or (a, b) == (v, u)), -1)]:
                    new_dist = d + c
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        queue.append((v, new_dist))
        
        # Print result
        if dist[y] == inf:
            print(-1)
        else:
            print(dist[y])