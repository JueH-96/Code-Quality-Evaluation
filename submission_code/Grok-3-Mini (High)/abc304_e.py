import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Union-Find setup
parent = [i for i in range(N + 1)]  # Index 0 not used

def find(parent, x):
    # Iterative find with path compression
    root = x
    while parent[root] != root:
        root = parent[root]
    # Compress path
    while x != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root

# Read M edges and perform union
for _ in range(M):
    u = int(data[index])
    index += 1
    v = int(data[index])
    index += 1
    rootU = find(parent, u)
    rootV = find(parent, v)
    if rootU != rootV:
        parent[rootU] = rootV  # Union by setting parent

# Compress all paths by calling find for each vertex
for i in range(1, N + 1):
    find(parent, i)  # After this, parent[x] is the root for each x

# Read K
K = int(data[index])
index += 1

# Set for forbidden component pairs
forbidden_set = set()

# Read K forbidden pairs and add to set
for _ in range(K):
    x = int(data[index])
    index += 1
    y = int(data[index])
    index += 1
    comp_x = parent[x]  # Component root
    comp_y = parent[y]  # Component root
    min_c = min(comp_x, comp_y)
    max_c = max(comp_x, comp_y)
    forbidden_set.add((min_c, max_c))

# Read Q
Q = int(data[index])
index += 1

# Process each query
for _ in range(Q):
    p = int(data[index])
    index += 1
    q = int(data[index])
    index += 1
    comp_p = parent[p]
    comp_q = parent[q]
    if comp_p == comp_q:
        # Same component, adding edge doesn't change connectivity
        print("Yes")
    else:
        # Different components, check if forbidden pair exists between them
        min_pq = min(comp_p, comp_q)
        max_pq = max(comp_p, comp_q)
        if (min_pq, max_pq) in forbidden_set:
            print("No")
        else:
            print("Yes")