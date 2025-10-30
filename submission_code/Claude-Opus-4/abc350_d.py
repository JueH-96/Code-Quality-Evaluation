# YOUR CODE HERE
def find_connected_components(n, edges):
    # Create adjacency list
    adj = [[] for _ in range(n + 1)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    visited = [False] * (n + 1)
    components = []
    
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)
    
    for i in range(1, n + 1):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    return components

# Read input
n, m = map(int, input().split())
edges = []
for _ in range(m):
    a, b = map(int, input().split())
    edges.append((a, b))

# Find connected components
components = find_connected_components(n, edges)

# Count existing edges in each component
edge_set = set()
for a, b in edges:
    edge_set.add((min(a, b), max(a, b)))

# Calculate maximum new friendships
total_new_friendships = 0

for component in components:
    size = len(component)
    if size <= 1:
        continue
    
    # Maximum possible edges in this component
    max_edges = size * (size - 1) // 2
    
    # Count existing edges in this component
    existing_edges = 0
    for i in range(len(component)):
        for j in range(i + 1, len(component)):
            a, b = component[i], component[j]
            if (min(a, b), max(a, b)) in edge_set:
                existing_edges += 1
    
    # New friendships that can be formed
    new_friendships = max_edges - existing_edges
    total_new_friendships += new_friendships

print(total_new_friendships)