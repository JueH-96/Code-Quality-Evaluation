def find_parent(parent, x):
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    px, py = find_parent(parent, x), find_parent(parent, y)
    if px == py:
        return
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1

n, m = map(int, input().split())

parent = list(range(n + 1))
rank = [0] * (n + 1)
edges = []

for _ in range(m):
    a, b = map(int, input().split())
    edges.append((a, b))
    union(parent, rank, a, b)

# Count nodes and edges in each component
component_nodes = {}
component_edges = {}

for i in range(1, n + 1):
    root = find_parent(parent, i)
    if root not in component_nodes:
        component_nodes[root] = 0
        component_edges[root] = 0
    component_nodes[root] += 1

for a, b in edges:
    root = find_parent(parent, a)
    component_edges[root] += 1

total_operations = 0
for root in component_nodes:
    nodes = component_nodes[root]
    current_edges = component_edges[root]
    max_edges = nodes * (nodes - 1) // 2
    total_operations += max_edges - current_edges

print(total_operations)