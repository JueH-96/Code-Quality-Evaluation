n = int(input())
edges = []
for _ in range(n):
    edges.append(input().strip())

from collections import defaultdict

# Build graph
in_deg = defaultdict(int)
out_deg = defaultdict(int)
vertices = set()

for edge in edges:
    u, v = edge[0], edge[1]
    out_deg[u] += 1
    in_deg[v] += 1
    vertices.add(u)
    vertices.add(v)

# Union-Find for weakly connected components
parent = {}

def find(x):
    if x not in parent:
        parent[x] = x
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

# Unite vertices connected by edges
for edge in edges:
    union(edge[0], edge[1])

# Group vertices by component
components = defaultdict(list)
for v in vertices:
    components[find(v)].append(v)

# Calculate minimum paths needed
total = 0
for comp in components.values():
    # Sum of positive excesses (out-degree - in-degree)
    pos_excess = 0
    for v in comp:
        excess = out_deg[v] - in_deg[v]
        if excess > 0:
            pos_excess += excess
    
    # Need at least 1 path per component
    total += max(1, pos_excess)

print(total)