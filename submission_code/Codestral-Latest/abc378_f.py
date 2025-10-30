import sys
from collections import defaultdict, deque

def count_valid_graphs(N, edges):
    # Build the tree
    tree = defaultdict(list)
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)

    # Find all leaves (vertices with degree 1)
    leaves = [i for i in range(1, N+1) if len(tree[i]) == 1]

    # BFS to find the depth of each leaf
    depth = [-1] * (N + 1)
    queue = deque(leaves)
    for leaf in leaves:
        depth[leaf] = 0

    while queue:
        node = queue.popleft()
        for neighbor in tree[node]:
            if depth[neighbor] == -1:
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)

    # Count the number of leaves at even and odd depths
    even_depth_leaves = sum(1 for d in depth if d % 2 == 0)
    odd_depth_leaves = sum(1 for d in depth if d % 2 == 1)

    # The number of valid graphs is the product of even and odd depth leaves
    return even_depth_leaves * odd_depth_leaves

# Read input
N = int(sys.stdin.readline().strip())
edges = []
for _ in range(N - 1):
    u, v = map(int, sys.stdin.readline().strip().split())
    edges.append((u, v))

# Calculate and print the result
result = count_valid_graphs(N, edges)
print(result)