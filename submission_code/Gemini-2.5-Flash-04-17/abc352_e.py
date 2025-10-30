import sys

# Increase recursion depth for DSU find with path compression.
# Maximum N is 2e5, recursion depth could theoretically go up to N
# before path compression is fully effective in some specific cases,
# though typically much less. Set a limit slightly larger than N.
sys.setrecursionlimit(200005)

def find(parent, i):
    """Find representative of the set containing element i."""
    if parent[i] == i:
        return i
    # Path compression
    parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, size, i, j):
    """
    Unite the sets containing elements i and j.
    Returns True if i and j were in different sets, False otherwise.
    Uses union by size heuristic.
    """
    root_i = find(parent, i)
    root_j = find(parent, j)

    if root_i != root_j:
        # Union by size: attach smaller tree to root of larger tree
        if size[root_i] < size[root_j]:
            root_i, root_j = root_j, root_i # swap roots

        parent[root_j] = root_i
        size[root_i] += size[root_j]
        return True # Successfully merged
    return False # Already in the same set

# Read input from stdin
# Use sys.stdin.readline for faster input in competitive programming
N, M = map(int, sys.stdin.readline().split())

operations = []
for _ in range(M):
    # Read K and C from the first line of the block
    line1 = list(map(int, sys.stdin.readline().split()))
    K = line1[0]
    C = line1[1]
    # Read the list of vertices S from the second line
    S = list(map(int, sys.stdin.readline().split()))
    operations.append((C, S)) # Store as (weight, vertex_list)

# Sort operations by weight (ascending order).
# This is crucial for Kruskal's-like approach.
operations.sort()

# Initialize DSU structure. Vertices are 1 to N.
# parent array: parent[i] is the parent of node i. Initially, each node is its own parent.
parent = list(range(N + 1))
# size array: size[i] is the size of the component if i is the root. Initially, each component has size 1.
size = [1] * (N + 1)

mst_weight = 0
edges_added_to_mst = 0 # Counts edges added to the MST forest

# Process operations in increasing order of weight
for C, S in operations:
    # For the current set S with weight C, find the components involved.
    # We only care about the representative of each component.
    unique_reps = set()
    for v in S:
        unique_reps.add(find(parent, v))

    # If the vertices in S belong to more than one component, we can add
    # edges of weight C to potentially connect these components.
    # To connect m components, we need m-1 edges.
    # By processing representatives and performing union, we effectively
    # add up to m-1 edges of weight C between these components.
    if len(unique_reps) > 1:
        # Pick an arbitrary representative from the set to merge others into.
        # set.pop() removes and returns an arbitrary element.
        first_rep = unique_reps.pop()

        # Attempt to union the remaining components' representatives with the first one.
        for rep in unique_reps:
            # If union is successful, it means we connected two previously
            # distinct components using an edge of weight C. This edge is
            # part of the MST because we are processing weights in increasing order.
            if union(parent, size, first_rep, rep):
                mst_weight += C
                edges_added_to_mst += 1

# After processing all operations, check if a spanning tree was formed.
# A graph with N vertices is connected if and only if its MST has N-1 edges.
if edges_added_to_mst == N - 1:
    # The graph is connected, print the total weight of the MST.
    print(mst_weight)
else:
    # The graph is not connected.
    print(-1)