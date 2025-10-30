from collections import deque, defaultdict

def bfs(graph, start, component_vertices):
    distances = {}
    queue = deque([start])
    distances[start] = 0
    
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v in component_vertices and v not in distances:
                distances[v] = distances[u] + 1
                queue.append(v)
    
    return distances

# Read input
N1, N2, M = map(int, input().split())

# Build adjacency list
graph = defaultdict(list)
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# Define components
first_component = set(range(1, N1 + 1))
second_component = set(range(N1 + 1, N1 + N2 + 1))

# BFS from vertex 1 in first component
dist_from_1 = bfs(graph, 1, first_component)

# BFS from vertex N1+N2 in second component
dist_from_last = bfs(graph, N1 + N2, second_component)

# Find maximum distance from 1 in first component
max_dist_from_1 = 0
for v in first_component:
    if v in dist_from_1:
        max_dist_from_1 = max(max_dist_from_1, dist_from_1[v])

# Find maximum distance from N1+N2 in second component
max_dist_from_last = 0
for v in second_component:
    if v in dist_from_last:
        max_dist_from_last = max(max_dist_from_last, dist_from_last[v])

# The answer is the sum of maximum distances plus 1 for the new edge
print(max_dist_from_1 + 1 + max_dist_from_last)