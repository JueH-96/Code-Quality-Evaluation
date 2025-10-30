# YOUR CODE HERE
def find_connected_components(n, edges):
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = [False] * (n + 1)
    components = 0
    
    def dfs(node):
        visited[node] = True
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    # Count connected components
    for i in range(1, n + 1):
        if not visited[i]:
            dfs(i)
            components += 1
    
    return components

# Read input
n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# Find number of connected components
k = find_connected_components(n, edges)

# Calculate minimum edges to delete
# A forest with n vertices and k components has n-k edges
# We need to delete m - (n-k) edges
min_deletions = m - (n - k)

print(min_deletions)