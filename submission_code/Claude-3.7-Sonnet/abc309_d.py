from collections import deque

def bfs(graph, start):
    dist = [-1] * len(graph)
    dist[start] = 0
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        for neighbor in graph[node]:
            if dist[neighbor] == -1:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    
    return dist

N1, N2, M = map(int, input().split())
graph = [[] for _ in range(N1 + N2 + 1)]

for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# Calculate distances from vertex 1 to all other vertices
dist_from_1 = bfs(graph, 1)

# Calculate distances from vertex N1+N2 to all other vertices
dist_from_N1_N2 = bfs(graph, N1 + N2)

# Find the vertex in the first group that's farthest from 1
max_dist_from_1 = 0
for u in range(1, N1 + 1):
    if dist_from_1[u] > max_dist_from_1:
        max_dist_from_1 = dist_from_1[u]

# Find the vertex in the second group that's farthest from N1+N2
max_dist_from_N1_N2 = 0
for v in range(N1 + 1, N1 + N2 + 1):
    if dist_from_N1_N2[v] > max_dist_from_N1_N2:
        max_dist_from_N1_N2 = dist_from_N1_N2[v]

# Add an edge between the two farthest vertices
result = max_dist_from_1 + 1 + max_dist_from_N1_N2
print(result)