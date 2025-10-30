from collections import deque

def bfs_distances(graph, start):
    n = len(graph)
    distances = [-1] * n
    distances[start] = 0
    queue = deque([start])
    
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if distances[v] == -1:
                distances[v] = distances[u] + 1
                queue.append(v)
    
    return distances

N1, N2, M = map(int, input().split())

# Build adjacency list (convert to 0-indexed)
graph = [[] for _ in range(N1 + N2)]
for _ in range(M):
    a, b = map(int, input().split())
    a -= 1  # Convert to 0-indexed
    b -= 1
    graph[a].append(b)
    graph[b].append(a)

# Calculate distances from vertex 1 (index 0) to all vertices
dist_from_1 = bfs_distances(graph, 0)

# Calculate distances from vertex N1+N2 (index N1+N2-1) to all vertices
dist_from_last = bfs_distances(graph, N1 + N2 - 1)

# Find maximum distance from vertex 1 to any vertex in first component
max_dist_1 = max(dist_from_1[u] for u in range(N1))

# Find maximum distance from any vertex in second component to vertex N1+N2
max_dist_last = max(dist_from_last[v] for v in range(N1, N1 + N2))

# The maximum possible distance is max_dist_1 + 1 + max_dist_last
max_distance = max_dist_1 + 1 + max_dist_last

print(max_distance)