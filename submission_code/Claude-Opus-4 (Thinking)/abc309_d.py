from collections import deque, defaultdict

def bfs(graph, start):
    """BFS to find shortest distances from start to all reachable vertices."""
    dist = {}
    dist[start] = 0
    queue = deque([start])
    
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    
    return dist

# Read input
N1, N2, M = map(int, input().split())

# Build adjacency list
graph = defaultdict(list)
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# BFS from vertex 1
dist_from_1 = bfs(graph, 1)

# BFS from vertex N1+N2
dist_from_N1N2 = bfs(graph, N1 + N2)

# Find the maximum distance
max_dist = 0
for u in range(1, N1 + 1):
    for v in range(N1 + 1, N1 + N2 + 1):
        dist = dist_from_1[u] + 1 + dist_from_N1N2[v]
        max_dist = max(max_dist, dist)

print(max_dist)