# YOUR CODE HERE
def solve():
    n = int(input())
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    edges = []
    
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
    
    # Calculate degrees
    degree = [0] * (n + 1)
    for i in range(1, n + 1):
        degree[i] = len(adj[i])
    
    # Find edges between degree-1 vertices (these were added during merging)
    added_edges = set()
    for u, v in edges:
        if degree[u] == 1 and degree[v] == 1:
            added_edges.add((min(u, v), max(u, v)))
    
    # Build graph without added edges
    original_adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        if (min(u, v), max(u, v)) not in added_edges:
            original_adj[u].append(v)
            original_adj[v].append(u)
    
    # Find connected components using DFS
    visited = [False] * (n + 1)
    components = []
    
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in original_adj[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)
    
    for i in range(1, n + 1):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    # Each component is a star, its level is size - 1
    levels = []
    for component in components:
        levels.append(len(component) - 1)
    
    # Sort and print
    levels.sort()
    print(' '.join(map(str, levels)))

solve()