# YOUR CODE HERE
import sys

# Increase recursion depth limit
# The maximum depth of recursion could be NK in a path graph
# NK <= 2 * 10^5, so 300000 should be sufficient.
sys.setrecursionlimit(300000)

# Function to read input efficiently
input = sys.stdin.readline

def solve():
    line1 = input().split()
    N = int(line1[0])
    K = int(line1[1])
    NK = N * K

    if K == 1:
        # If K=1, each path is a single vertex. We need to partition NK vertices into N paths of 1 vertex.
        # This is always possible. The graph structure is irrelevant for K=1.
        # We still need to read the input edges to consume them.
        for _ in range(NK - 1):
            input() # Read and discard edge lines
        print("Yes")
        return

    # Handle case NK=1 which implies N=1, K=1. Already covered by K=1 case.
    # If NK > 1 and K > 1, then NK-1 edges exist.
    # If NK=0, constraints say 1 <= N, 1 <= K, so NK >= 1.

    adj = [[] for _ in range(NK + 1)]
    for _ in range(NK - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # dfs(u, p) returns:
    # - L (0 < L < K): the length of the path fragment ending at u that needs to connect to parent p.
    #   This fragment uses L vertices from the subtree rooted at u (including u)
    #   and is rooted somewhere deeper in the subtree.
    # - 0: the subtree rooted at u (excluding edge to p) is perfectly decomposed into paths of length K.
    #   This includes the possibility that a path of length K ends exactly at u.
    # - -1: impossible decomposition in the subtree.
    def dfs(u, p):
        fragments_lengths_gt_0 = [] # Stores lengths L > 0 from children

        for v in adj[u]:
            if v == p:
                continue
            
            len_v = dfs(v, u)
            
            if len_v == -1:
                return -1 # Failure in child subtree

            if len_v > 0:
                fragments_lengths_gt_0.append(len_v)

        # After visiting all children:
        if len(fragments_lengths_gt_0) > 1:
            # Multiple children returned fragments needing u to extend. Impossible.
            return -1

        if len(fragments_lengths_gt_0) == 1:
            # u must extend the path fragment from the chosen child.
            child_fragment_len = fragments_lengths_gt_0[0]
            len_u = child_fragment_len + 1
            
            if len_u == K:
                # This forms a complete path of length K ending at u.
                # The subtree at u (excluding edge to p) is now perfectly decomposed.
                # Return 0 to signify perfect decomposition below u.
                return 0
            else: # 0 < len_u < K
                # This fragment of length len_u ends at u and needs to extend upwards to p.
                # Return its length.
                return len_u

        # len(fragments_lengths_gt_0) == 0
        # All children subtrees were perfectly decomposed (returned 0).
        # u starts a new path fragment of length 1, consisting of just u.
        len_u = 1
        # We already handled K=1 case at the beginning where len_u = 1 == K returns 0.
        # Here, K > 1, so len_u = 1 < K.
        # This fragment of length 1 ending at u needs to extend upwards to p.
        return len_u

    # Start DFS from an arbitrary root, say vertex 1. Parent of root is 0.
    # For the entire tree to be decomposed into paths of length K,
    # the call from the root must indicate perfect decomposition below the root.
    # This means dfs(root, 0) must return 0.
    
    # The root node should be 1 as per problem constraints (vertices 1 to NK).
    result = dfs(1, 0)

    if result == 0:
        print("Yes")
    else:
        print("No")

solve()