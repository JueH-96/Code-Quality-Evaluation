import sys

# Read N
N = int(sys.stdin.readline())

# Store weights
# weights[(i, j)] stores weight of edge between 0-indexed vertices i and j, where i < j
weights = {}
for i in range(N - 1):
    # Read the weights for edges starting from vertex i (0-indexed)
    # This corresponds to D_{i+1, j} for j from i+2 to N (1-indexed)
    # which means edges (i, j) for j from i+1 to N-1 (0-indexed)
    line = list(map(int, sys.stdin.readline().split()))
    for k in range(len(line)):
        j = i + 1 + k # Calculate the other vertex index (0-indexed)
        weights[(i, j)] = line[k]

# DP state: dp[mask] is the maximum weight matching for the vertices in mask
# mask is an integer where the k-th bit is set if vertex k is in the subset.
# Initialize with 0, as minimum edge weight is 1 and 0 is achievable (no edges).
dp = [0] * (1 << N)

# Iterate through all possible subsets of vertices (represented by masks)
# Masks are processed in increasing order, ensuring that when we compute dp[mask],
# the dp values for all proper subsets of mask have already been computed.
for mask in range(1, 1 << N):
    # Find the index u of the first set bit (lowest index vertex) in the mask.
    # Any maximum weight matching for the vertices in 'mask' either includes
    # an edge involving vertex 'u' or it does not.
    u = -1
    for i in range(N):
        if (mask >> i) & 1:
            u = i
            break # Found the lowest indexed vertex 'u' in the current mask

    # Option 1: Vertex u is NOT matched in the optimal matching for 'mask'.
    # In this case, the optimal matching for 'mask' is the same as the optimal
    # matching for the set of vertices 'mask' excluding 'u'.
    # This subset is represented by the mask 'mask ^ (1 << u)'.
    # Since mask > 0, u is found, and mask ^ (1 << u) is always a smaller mask.
    # We set the initial maximum for dp[mask] based on this option.
    dp[mask] = dp[mask ^ (1 << u)]

    # Option 2: Vertex u IS matched with some other vertex v (where v > u) in the mask.
    # We iterate through all possible partners 'v' for 'u' from the vertices
    # present in the mask that have an index greater than 'u'.
    for v in range(u + 1, N):
        # Check if vertex v is in the current mask
        if (mask >> v) & 1:
            # If u is matched with v, the edge (u, v) is part of the matching.
            # The remaining vertices for the matching are represented by 'mask'
            # with bits u and v removed: 'mask ^ (1 << u) ^ (1 << v)'.
            # The total weight would be the weight of edge (u, v) plus the maximum
            # weight matching on the remaining vertices.
            remaining_mask = mask ^ (1 << u) ^ (1 << v)
            current_weight = weights[(u, v)] + dp[remaining_mask]

            # Update dp[mask] with the maximum weight found so far, considering
            # matching u with v or not matching u at all (Option 1 value already set).
            dp[mask] = max(dp[mask], current_weight)

# The maximum weight matching for the entire graph (all vertices 0 to N-1)
# is stored in dp[(1 << N) - 1], where (1 << N) - 1 is the mask
# with all bits set, representing the set {0, 1, ..., N-1}.
print(dp[(1 << N) - 1])