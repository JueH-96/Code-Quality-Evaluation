# YOUR CODE HERE
import sys

# Increase recursion depth limit for potentially deep trees
# N can be up to 10^5, so the depth of DFS can be close to N.
# A limit of 300000 is safely more than max N.
sys.setrecursionlimit(300000)

# Read N
N = int(sys.stdin.readline())

# Build adjacency list (0-indexed)
adj = [[] for _ in range(N)]
# A tree with N vertices has N-1 edges.
# If N=1, the loop range(0) is correct and skips the edge reading.
for _ in range(N - 1):
    u, v = map(int, sys.stdin.readline().split())
    # Adjust to 0-indexed
    u -= 1
    v -= 1
    adj[u].append(v)
    adj[v].append(u)

# Read C values (0-indexed)
C = list(map(int, sys.stdin.readline().split()))

# Calculate Total_C: Sum of all C_i values.
# This is used in the transition formula between f values of adjacent nodes.
Total_C = sum(C)

# Pass 1: Calculate depth and subtree_weight using DFS from an arbitrary root (vertex 0).
# depth[i]: distance from the chosen root (vertex 0) to vertex i.
# subtree_weight[i]: sum of C values for all nodes in the subtree rooted at i
# (when the tree is rooted at vertex 0). This sum includes C[i].
depth = [0] * N
subtree_weight = [0] * N

def dfs1(u, parent, current_depth):
    """
    First DFS pass to calculate depth and subtree_weight for each node.
    u: current node (0-indexed)
    parent: parent of u in the DFS tree (-1 for the root)
    current_depth: depth of u from the root (vertex 0)
    """
    depth[u] = current_depth
    # Initialize subtree_weight[u] with the node's own weight C[u].
    subtree_weight[u] = C[u]
    # Recursively call for children and accumulate subtree weights.
    for v in adj[u]:
        if v != parent:
            dfs1(v, u, current_depth + 1)
            # Add the subtree weight of the child v to the subtree weight of u.
            subtree_weight[u] += subtree_weight[v]

# Start the first DFS from root 0 (vertex 1). Parent of root is -1. Initial depth is 0.
# This call is valid even for N=1, as adj[0] will be empty.
dfs1(0, -1, 0)

# Calculate f(root=0): The total weighted distance from vertex 0 to all other nodes.
# f(0) = sum_{i=0 to N-1} (C[i] * d(0, i)).
# In the DFS tree rooted at 0, d(0, i) is simply depth[i].
f_root = 0
# This loop is valid even for N=1 (range(1)).
for i in range(N):
    f_root += C[i] * depth[i]

# Pass 2: Calculate f_values for all nodes using a second DFS.
# We use the relationship: f(v) = f(u) + Total_C - 2 * subtree_weight[v]
# where u is the parent of v in the second DFS traversal, and subtree_weight[v]
# is the sum calculated in the first DFS (sum of C in the subtree rooted at v when
# the tree is rooted at 0).
f_values = [0] * N

def dfs2(u, parent, f_u):
    """
    Second DFS pass to calculate f_values for all nodes using the root's f_value
    and the precomputed subtree weights.
    u: current node (0-indexed)
    parent: parent of u in the current DFS traversal (-1 for the initial call)
    f_u: the calculated f value for node u
    """
    # Store the calculated f value for node u.
    f_values[u] = f_u
    # Recursively call for neighbors (which will be children in this traversal,
    # except for the parent).
    # This loop is valid even for N=1, as adj[0] will be empty.
    for v in adj[u]:
        if v != parent:
            # Calculate f(v) using f(u) and subtree_weight[v].
            # Moving from u to v:
            # Nodes in v's subtree (rooted at 0) get 1 unit closer to v than u.
            # Nodes outside v's subtree get 1 unit farther from v than u.
            # Sum of C in v's subtree = subtree_weight[v] (from dfs1)
            # Sum of C outside v's subtree = Total_C - subtree_weight[v]
            # f(v) = f(u) + (Sum C outside v's subtree) - (Sum C inside v's subtree)
            f_v = f_u + (Total_C - subtree_weight[v]) - subtree_weight[v]
            # Simplified: f_v = f_u + Total_C - 2 * subtree_weight[v]
            # Continue DFS from v. u is now the parent of v in this traversal.
            dfs2(v, u, f_v)

# Start the second DFS from root 0, using the calculated f_root. Parent of root is -1.
dfs2(0, -1, f_root)

# The answer is the minimum value among all calculated f_values.
# f_values[i] corresponds to f(vertex i+1).
# If N=1, f_values = [0], min is 0.
min_f = min(f_values)

# Print the minimum f value.
print(min_f)