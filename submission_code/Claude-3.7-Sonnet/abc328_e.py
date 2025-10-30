from itertools import combinations

def is_spanning_tree(n, selected_edges):
    """Check if the given edges form a spanning tree."""
    parent = list(range(n + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    # Check if the selected edges form a tree (no cycles)
    for u, v, _ in selected_edges:
        if find(u) == find(v):
            return False  # Cycle detected
        union(u, v)
    
    # Check if all vertices are in the same component
    for i in range(2, n + 1):
        if find(i) != find(1):
            return False
    
    return True

def min_cost_spanning_tree(n, m, k, edges):
    min_cost = float('inf')
    
    for selected_edges in combinations(edges, n - 1):
        if is_spanning_tree(n, selected_edges):
            cost = sum(w for _, _, w in selected_edges) % k
            min_cost = min(min_cost, cost)
    
    return min_cost

# Read input
n, m, k = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

print(min_cost_spanning_tree(n, m, k, edges))