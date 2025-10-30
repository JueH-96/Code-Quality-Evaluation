n, m = map(int, input().split())

# Build adjacency list
adj = [[] for _ in range(n + 1)]
for _ in range(m):
    a, b = map(int, input().split())
    adj[a].append(b)
    adj[b].append(a)

# Find connected components using DFS
visited = [False] * (n + 1)
component_sizes = []

def dfs(v):
    visited[v] = True
    size = 1
    for u in adj[v]:
        if not visited[u]:
            size += dfs(u)
    return size

for i in range(1, n + 1):
    if not visited[i]:
        size = dfs(i)
        component_sizes.append(size)

# Calculate total edges in the final state
total_final_edges = 0
for size in component_sizes:
    # A complete graph with 'size' vertices has size*(size-1)/2 edges
    total_final_edges += size * (size - 1) // 2

# The answer is the difference between final and initial edges
answer = total_final_edges - m
print(answer)