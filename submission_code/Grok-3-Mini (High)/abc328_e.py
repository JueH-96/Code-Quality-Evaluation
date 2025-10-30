import sys
import itertools

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
K = int(data[index])
index += 1

# Read edges, make vertices 0-based
edges = []
for _ in range(M):
    u = int(data[index])
    index += 1
    v = int(data[index])
    index += 1
    w = int(data[index])
    index += 1
    edges.append((u-1, v-1, w))

# Define the function to check if the subset is a spanning tree
def is_spanning_tree(edge_indices, edges, N):
    parent = [i for i in range(N)]
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    # Union the edges
    for idx in edge_indices:
        u, v, _ = edges[idx]
        rootU = find(u)
        rootV = find(v)
        if rootU != rootV:
            parent[rootU] = rootV  # Union them
    
    # Check if all nodes have the same root
    root0 = find(0)
    for i in range(1, N):
        if find(i) != root0:
            return False
    return True

# Now, iterate over all combinations of N-1 edges
min_cost = None
for comb in itertools.combinations(range(M), N-1):
    if is_spanning_tree(comb, edges, N):
        # Compute sum of weights
        sum_w = sum(edges[idx][2] for idx in comb)
        cost = sum_w % K
        if min_cost is None or cost < min_cost:
            min_cost = cost

# Output the minimum cost
print(min_cost)