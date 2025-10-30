import sys
import collections
import math

# Increase recursion depth for potentially deep DFS
# N up to 2e5, depth could be N. Default limit might be 1000.
# 300000 should be sufficient.
sys.setrecursionlimit(300000)

# Constants
# N up to 2 * 10^5. log2(2 * 10^5) approx 17.6. We need up to 2^k >= N-1 distance.
# Maximum depth is N-1. The 2^k-th ancestor could be needed.
# 2^17 = 131072. 2^18 = 262144.
# LOGN_MAX = 18 means k goes up to 17 (0-indexed). This should be sufficient for parents up to 2^17 steps away.
# To jump 2^k steps, we need parent[u][k]. For parent[u][k] = parent[parent[u][k-1]][k-1], the max k is LOGN_MAX - 1.
# So parent array needs indices up to LOGN_MAX - 1. Size is LOGN_MAX.
LOGN_MAX = 18 # Number of columns in parent table

# Globals for tree data
adj = collections.defaultdict(list)
depth = []
parent = []
tin = []
tout = []
timer = 0
N_nodes = 0 # Use N_nodes to avoid conflict with input variable N

def dfs(u, p, d):
    """Performs DFS to compute depth, parent pointers, and DFS entry/exit times."""
    global timer
    tin[u] = timer
    timer += 1
    depth[u] = d
    parent[u][0] = p

    # Precompute 2^k-th ancestors using binary lifting
    for k in range(1, LOGN_MAX):
        if parent[u][k-1] != 0:
            parent[u][k] = parent[parent[u][k-1]][k-1]
        else:
            parent[u][k] = 0 # Sentinel for no 2^k-th ancestor

    # Recurse on children
    for v in adj[u]:
        if v != p:
            dfs(v, u, d + 1)

    tout[u] = timer
    timer += 1

def is_ancestor(u, v):
    """Checks if node u is an ancestor of node v using DFS entry and exit times."""
    # Assumes u and v are valid vertex indices (1 to N_nodes) or sentinel 0.
    # Sentinel 0 (parent of root) is considered ancestor of any valid node (1 to N_nodes)
    if u == 0:
        return True
    # A valid node (1 to N_nodes) cannot be an ancestor of the sentinel 0
    if v == 0:
        return False
    # For two valid nodes, u is ancestor of v if u was entered before v and exited after v
    return tin[u] <= tin[v] and tout[u] >= tout[v]

def lca(u, v):
    """Computes the Lowest Common Ancestor of nodes u and v."""
    # Ensure u and v are valid nodes (1 to N_nodes)
    # The function is designed to be called with valid node indices from required_vertices.
    # During the lifting process, u might become the sentinel 0.
    if u == 0: return v # Should not happen with valid initial calls
    if v == 0: return u # Should not happen with valid initial calls

    # If one node is an ancestor of the other, that ancestor is the LCA
    if is_ancestor(u, v):
        return u
    if is_ancestor(v, u):
        return v

    # Lift u up using binary lifting
    # Start from the largest jump (2^(LOGN_MAX-1)) down to 2^0
    for k in range(LOGN_MAX - 1, -1, -1):
        # Try to jump u up by 2^k steps
        # The ancestor parent[u][k] must be valid (not 0) AND
        # parent[u][k] must NOT be an ancestor of v (because if it is, we might have jumped too high past the LCA)
        if parent[u][k] != 0 and not is_ancestor(parent[u][k], v):
            u = parent[u][k]

    # After the loop, u is the deepest node that is an ancestor of the original u but *not* an ancestor of v.
    # This means parent[u][0] (the immediate parent of this final u) must be the LCA of the original u and v.
    return parent[u][0]

def solve():
    """Reads input, computes the minimum number of vertices, and prints the result."""
    global N_nodes, depth, parent, tin, tout, adj, timer
    N_nodes, K = map(int, sys.stdin.readline().split())

    # Reset adjacency list for the current test case
    adj = collections.defaultdict(list)
    for _ in range(N_nodes - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Initialize data structures (1-based indexing for vertices 1..N_nodes)
    depth = [0] * (N_nodes + 1)
    parent = [[0] * LOGN_MAX for _ in range(N_nodes + 1)]
    tin = [0] * (N_nodes + 1)
    tout = [0] * (N_nodes + 1)

    # Perform DFS from root 1 (arbitrary choice). Depth of root is 0.
    timer = 0
    dfs(1, 0, 0) # Root 1, parent 0 (sentinel), depth 0

    # Read the K specified vertices
    required_vertices = list(map(int, sys.stdin.readline().split()))

    # If only one vertex is required, the minimal subtree is just that vertex itself, which has 1 vertex.
    if K == 1:
        print(1)
        return

    # Sort the required vertices based on their discovery time (tin) from the DFS traversal.
    # This ordering is necessary for the formula involving the sum of distances between adjacent pairs in the sorted list.
    required_vertices.sort(key=lambda v: tin[v])

    # Calculate the sum of depths of all required vertices.
    # The depth of a vertex is the number of edges on the path from the root (vertex 1) to that vertex.
    total_depth_sum = sum(depth[v] for v in required_vertices)

    # Calculate the sum of depths of the LCAs of adjacent pairs of required vertices
    # in the DFS-sorted order, including the cyclic pair (last, first).
    # This sum is part of the formula for the number of edges in the minimal subtree.
    lca_depths_sum = 0
    for i in range(K):
        # Consider adjacent vertices in the sorted list, wrapping around from the last to the first
        u = required_vertices[i]
        v = required_vertices[(i + 1) % K]
        common_ancestor = lca(u, v)
        lca_depths_sum += depth[common_ancestor]

    # The number of edges in the minimal subtree containing all required vertices
    # is given by (Sum of depths of required vertices) - (Sum of depths of LCAs of adjacent required vertices in DFS order tour).
    # This formula arises from the total edge length of the Euler tour that visits all required vertices
    # in DFS order and returns to the start. This tour traverses each edge of the minimal subtree exactly twice.
    # The length of the tour = sum of distances between adjacent vertices in the sorted sequence (including wrapping).
    # distance(u, v) = depth(u) + depth(v) - 2 * depth(LCA(u, v)) (number of edges).
    # Sum of distances = sum(depth(v_i) + depth(v_{i+1}) - 2*depth(LCA(v_i, v_{i+1}))) over cyclic adjacent pairs.
    # Sum of distances = 2 * sum(depth(v) for v in required_vertices) - 2 * sum(depth(LCA(v_i, v_{i+1}))) over cyclic adjacent pairs.
    # Number of edges = (Sum of distances) / 2 = sum(depth(v)) - sum(depth(LCA(v_i, v_{i+1}))).

    # Therefore, the number of edges in the minimal subtree is total_depth_sum - lca_depths_sum.
    num_edges = total_depth_sum - lca_depths_sum

    # The number of vertices in a tree is always one more than the number of edges.
    num_vertices = num_edges + 1

    # Print the calculated minimum number of vertices.
    print(num_vertices)

# The script should read from stdin and write to stdout as per the format.
# Ensure no extra print statements or test cases are included.
if __name__ == "__main__":
    solve()