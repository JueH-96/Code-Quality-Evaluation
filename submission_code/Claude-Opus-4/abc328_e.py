# YOUR CODE HERE
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    px, py = find(parent, x), find(parent, y)
    if px == py:
        return False
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1
    return True

def is_spanning_tree(n, edges):
    if len(edges) != n - 1:
        return False
    
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    for u, v, _ in edges:
        if not union(parent, rank, u, v):
            return False
    
    # Check if all vertices are connected
    root = find(parent, 1)
    for i in range(2, n + 1):
        if find(parent, i) != root:
            return False
    
    return True

n, m, k = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

min_cost = float('inf')

# Try all combinations of n-1 edges
from itertools import combinations

for edge_subset in combinations(range(m), n - 1):
    selected_edges = [edges[i] for i in edge_subset]
    
    if is_spanning_tree(n, selected_edges):
        cost = sum(edge[2] for edge in selected_edges) % k
        min_cost = min(min_cost, cost)

print(min_cost)