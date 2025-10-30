from itertools import combinations

def is_spanning_tree(edges, n):
    """Check if the given edges form a spanning tree with n vertices"""
    if len(edges) != n - 1:
        return False
    
    if n == 1:
        return len(edges) == 0
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    vertices_in_graph = set()
    
    for u, v, w in edges:
        adj[u].append(v)
        adj[v].append(u)
        vertices_in_graph.add(u)
        vertices_in_graph.add(v)
    
    # Check if all vertices from 1 to n are included
    if vertices_in_graph != set(range(1, n + 1)):
        return False
    
    # DFS to check connectivity
    visited = [False] * (n + 1)
    stack = [1]
    visited[1] = True
    count = 1
    
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
                count += 1
    
    return count == n

n, m, k = map(int, input().split())
edges = []

for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

min_cost = float('inf')

# Try all combinations of n-1 edges
for edge_subset in combinations(range(m), n - 1):
    selected_edges = [edges[i] for i in edge_subset]
    
    if is_spanning_tree(selected_edges, n):
        cost = sum(w for u, v, w in selected_edges) % k
        min_cost = min(min_cost, cost)
        if min_cost == 0:
            break

print(min_cost)