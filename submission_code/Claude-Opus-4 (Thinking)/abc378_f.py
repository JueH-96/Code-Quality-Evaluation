from collections import deque

def find_path(graph, start, end, n):
    # BFS to find path
    parent = [-1] * (n + 1)
    visited = [False] * (n + 1)
    queue = deque([start])
    visited[start] = True
    
    while queue:
        u = queue.popleft()
        if u == end:
            break
        for v in graph[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)
    
    # Reconstruct path
    path = []
    curr = end
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    return path

n = int(input())
graph = [[] for _ in range(n + 1)]
degree = [0] * (n + 1)

for _ in range(n - 1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)
    degree[u] += 1
    degree[v] += 1

# Find all vertices with degree 2
degree_2_vertices = []
for i in range(1, n + 1):
    if degree[i] == 2:
        degree_2_vertices.append(i)

count = 0
# Check all pairs of degree-2 vertices
for i in range(len(degree_2_vertices)):
    for j in range(i + 1, len(degree_2_vertices)):
        u = degree_2_vertices[i]
        v = degree_2_vertices[j]
        
        # Check if u and v are adjacent (would create multi-edge)
        if v in graph[u]:
            continue
        
        # Find path from u to v
        path = find_path(graph, u, v, n)
        
        # Check if all intermediate vertices have degree 3
        valid = True
        for k in range(1, len(path) - 1):
            if degree[path[k]] != 3:
                valid = False
                break
        
        if valid:
            count += 1

print(count)