from collections import deque

def bfs(graph, start):
    n = len(graph)
    dist = [-1] * n
    dist[start] = 0
    queue = deque([start])
    
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    
    return dist

# Read input
N1, N2, M = map(int, input().split())

# Build adjacency list
graph = [[] for _ in range(N1 + N2)]
for _ in range(M):
    a, b = map(int, input().split())
    a -= 1  # Convert to 0-indexed
    b -= 1
    graph[a].append(b)
    graph[b].append(a)

# Find distances from vertex 1 (index 0) in first component
dist_from_1 = bfs(graph, 0)

# Find distances from vertex N1+N2 (index N1+N2-1) in second component
dist_from_end = bfs(graph, N1 + N2 - 1)

# Find maximum possible distance
max_dist = 0

# Try all possible edges between the two components
for u in range(N1):  # vertices in first component
    for v in range(N1, N1 + N2):  # vertices in second component
        # The shortest path would be: dist_from_1[u] + 1 + dist_from_end[v]
        path_length = dist_from_1[u] + 1 + dist_from_end[v]
        max_dist = max(max_dist, path_length)

print(max_dist)