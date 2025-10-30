from itertools import combinations

def is_connected(n, edges):
    # Check if the given edges form a connected graph with n vertices
    if len(edges) != n - 1:
        return False
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for u, v, w in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # DFS to check connectivity
    visited = [False] * (n + 1)
    
    def dfs(node):
        visited[node] = True
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    # Start DFS from vertex 1
    dfs(1)
    
    # Check if all vertices (1 to n) are visited
    for i in range(1, n + 1):
        if not visited[i]:
            return False
    
    return True

# Read input
n, m, k = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

min_cost = float('inf')

# Try all combinations of n-1 edges
for edge_combination in combinations(edges, n - 1):
    if is_connected(n, edge_combination):
        # Calculate cost modulo k
        cost = sum(w for u, v, w in edge_combination) % k
        min_cost = min(min_cost, cost)

print(min_cost)