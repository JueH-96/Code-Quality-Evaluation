n, m = map(int, input().split())

# Build adjacency list
adj = [[] for _ in range(n+1)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

# Find number of connected components using DFS
visited = [False] * (n+1)
components = 0

def dfs(node):
    visited[node] = True
    for neighbor in adj[node]:
        if not visited[neighbor]:
            dfs(neighbor)

for i in range(1, n+1):
    if not visited[i]:
        dfs(i)
        components += 1

# A forest with n vertices and k components can have at most n-k edges
# So we need to delete max(0, m - (n - components)) edges
print(max(0, m - n + components))