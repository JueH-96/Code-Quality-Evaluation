# NUMBER OF VERTICES AND SPECIFIED VERTICES
N, K = map(int, input().split())

# TREE EDGES
edges = []
for _ in range(N-1):
    A, B = map(int, input().split())
    edges.append((min(A, B), max(A, B)))

# SPECIFIED VERTICES
V = list(map(int, input().split()))

# FINDING THE CONNECTED COMPONENTS THAT CONTAIN SPECIFIED VERTICES
connected_components = []
visited = [False] * (N+1)
for v in V:
    if not visited[v]:
        component = []
        stack = [v]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                component.append(node)
                for a, b in edges:
                    if a == node and not visited[b]:
                        stack.append(b)
                    elif b == node and not visited[a]:
                        stack.append(a)
        connected_components.append(component)

# FINDING THE MINIMUM NUMBER OF VERTICES IN THE CONNECTED COMPONENTS
min_vertices = len(V)
for i in range(len(connected_components)):
    for j in range(i+1, len(connected_components)):
        common_vertices = set(connected_components[i]) & set(connected_components[j])
        if common_vertices:
            min_vertices = min(min_vertices, len(connected_components[i]) + len(connected_components[j]) - len(common_vertices))

print(min_vertices)