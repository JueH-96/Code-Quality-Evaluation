import sys

# DSU implementation (0-indexed)
parent = []
sz = []

def initialize_dsu(n):
    """Initializes the DSU structure for n elements (0 to n-1)."""
    global parent, sz
    parent = list(range(n))
    sz = [1] * n

def find(i):
    """Finds the root of the set containing element i with path compression."""
    # Iterative find with path compression
    root = i
    while root != parent[root]:
        root = parent[root]
    # Path compression: make all nodes on the path point directly to the root
    curr = i
    while curr != parent[curr]:
        next_node = parent[curr]
        parent[curr] = root
        curr = next_node
    return root

def union(i, j):
    """Unites the sets containing elements i and j using union by size.
       Returns True if a merge occurred, False otherwise."""
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        # Union by size: attach smaller tree to larger tree
        if sz[root_i] < sz[root_j]:
            root_i, root_j = root_j, root_i
        parent[root_j] = root_i
        sz[root_i] += sz[root_j]
        return True # Components were merged
    return False # Components were already the same (edge created a cycle)

# Read N and M
N, M = map(int, sys.stdin.readline().split())

# Initialize DSU for N vertices (0 to N-1)
initialize_dsu(N)

# Count edges that form cycles
# These are the edges that connect two vertices already in the same connected component.
# To make the graph a forest (acyclic), we must remove at least one edge from each cycle.
# The minimum number of edges to remove is equal to the number of fundamental cycles,
# which corresponds to the number of edges that close a cycle when processed sequentially.
edges_to_delete = 0

# Process edges
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    # Convert 1-indexed vertices to 0-indexed
    u0 = u - 1
    v0 = v - 1

    # Attempt to union the components containing u and v.
    # If union returns False, it means u and v were already in the same component,
    # so adding this edge created a cycle.
    if not union(u0, v0):
        edges_to_delete += 1

# Print the result
print(edges_to_delete)