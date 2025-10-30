from collections import defaultdict, deque

n = int(input())

# Build the tree
graph = defaultdict(list)
degree = [0] * (n + 1)

for _ in range(n - 1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)
    degree[u] += 1
    degree[v] += 1

def find_path(start, end):
    if start == end:
        return [start]
    
    parent = [-1] * (n + 1)
    visited = [False] * (n + 1)
    queue = deque([start])
    visited[start] = True
    
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                queue.append(neighbor)
    
    # Reconstruct path
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = parent[current]
    
    return path[::-1]

# Find all vertices with degree 2
degree_2_vertices = []
for i in range(1, n + 1):
    if degree[i] == 2:
        degree_2_vertices.append(i)

count = 0

for i in range(len(degree_2_vertices)):
    for j in range(i + 1, len(degree_2_vertices)):
        a, b = degree_2_vertices[i], degree_2_vertices[j]
        
        # Check if a and b are already connected
        if b in graph[a]:
            continue
        
        # Find path from a to b
        path = find_path(a, b)
        
        # Check if all intermediate vertices have degree 3
        valid = True
        for k in range(1, len(path) - 1):  # Exclude endpoints
            if degree[path[k]] != 3:
                valid = False
                break
        
        if valid:
            count += 1

print(count)