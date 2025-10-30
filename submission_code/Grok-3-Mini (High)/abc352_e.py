import sys

def find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    # Compress path
    current = x
    while current != root:
        next_p = parent[current]
        parent[current] = root
        current = next_p
    return root

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Read M operations
operations = []
for _ in range(M):
    K = int(data[index])
    index += 1
    C_weight = int(data[index])  # weight
    index += 1
    vertices = []
    for _ in range(K):
        vert = int(data[index])
        index += 1
        vertices.append(vert)
    operations.append((C_weight, vertices))

# Sort operations by weight ascending
operations.sort(key=lambda x: x[0])

# Initialize Union-Find
parent = [i for i in range(N + 1)]
rank = [0] * (N + 1)
comp_count = N
cost = 0

# Process each operation in sorted order
for weight, verts in operations:
    # Find the set of roots for verts
    roots_set = set()
    for v in verts:
        root_v = find(parent, v)
        roots_set.add(root_v)
    C = len(roots_set)
    if C >= 2:
        cost += (C - 1) * weight
        comp_count -= (C - 1)
        # Union all roots to the first root
        if roots_set:
            roots_list = list(roots_set)
            first_root = roots_list[0]
            for other_root in roots_list[1:]:
                union(parent, rank, first_root, other_root)

# Check connectivity and output result
if comp_count == 1:
    print(cost)
else:
    print(-1)