# YOUR CODE HERE
def is_spanning_tree(edges, n):
    if len(edges) != n - 1:
        return False
    
    parent = list(range(n + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False  # Cycle detected
        parent[px] = py
        return True
    
    for u, v, _ in edges:
        if not union(u, v):
            return False
    
    # Check if all vertices are connected
    components = set()
    for i in range(1, n + 1):
        components.add(find(i))
    
    return len(components) == 1

n, m, k = map(int, input().split())
edges = []

for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

min_cost = k  # Initialize to k (any valid cost will be < k)

from itertools import combinations

for subset in combinations(range(m), n - 1):
    edge_subset = [edges[i] for i in subset]
    if is_spanning_tree(edge_subset, n):
        cost = sum(e[2] for e in edge_subset) % k
        min_cost = min(min_cost, cost)

print(min_cost)