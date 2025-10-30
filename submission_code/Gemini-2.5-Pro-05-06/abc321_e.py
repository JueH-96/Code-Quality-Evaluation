import sys

def solve():
    N, X, K = map(int, sys.stdin.readline().split())

    if K == 0:
        sys.stdout.write("1
")
        return

    ans = 0
    
    # N_bit_len stores N.bit_length() for the current N.
    # This is used in get_nodes_at_level for an important optimization.
    N_bit_len = N.bit_length()

    def get_nodes_at_level(u, d, N_limit):
        # Counts nodes in u's subtree at relative depth d, which are <= N_limit.
        
        if u == 0: # Not a valid node
            return 0 
        if u > N_limit: # Base node u itself is out of bounds
            return 0 
        if d < 0: # Relative depth cannot be negative
            return 0
        
        # Optimization:
        # N_limit < 2^(N_limit.bit_length()).
        # If d >= N_limit.bit_length(), then 2^d >= 2^(N_limit.bit_length()) > N_limit.
        # So, for u >= 1, the smallest node L = u * 2^d would be > N_limit.
        # Hence, no nodes at this level are <= N_limit.
        if d >= N_bit_len : 
             return 0

        # m = 2^d
        # Python's 1 << d handles large d by creating large integers.
        # Max d here is N_bit_len - 1, around 59-60 for N up to 10^18.
        # 2^59 is large but manageable.
        m = 1 << d
        
        # Smallest node value in the full binary tree at this level in u's subtree
        low = u * m
        
        # If smallest node is already too large, no nodes are valid.
        if low > N_limit: 
            return 0
            
        # Largest node value in the full binary tree at this level in u's subtree
        high = low + m - 1 
        
        # Count nodes in [low, high] that are also <= N_limit
        return min(N_limit, high) - low + 1

    # Case 1: v is an ancestor of X.
    # The K-th ancestor of X is X // (2^K), or X >> K.
    # It must be a valid node ID (>= 1).
    kth_ancestor = X >> K
    if kth_ancestor >= 1:
        ans += 1
    
    # Case 2: v is a descendant of X.
    # These are nodes in X's subtree at relative depth K.
    ans += get_nodes_at_level(X, K, N)

    # Case 3: LCA(X,v) is a proper ancestor of X.
    # Let dist(X, LCA) = k_up and dist(v, LCA) = k_down.
    # k_up + k_down = K.
    # k_up ranges from 1 to K-1.
    # Max value for k_up is also limited by depth of X. If X >> k_up is 0, we stop.
    for k_up in range(1, K): 
        LCA = X >> k_up 
        if LCA == 0: # Current LCA candidate is above root, so no more valid LCAs.
            break 
        
        # child_on_X_path is the child of LCA that is an ancestor of X (or X itself).
        child_on_X_path = X >> (k_up - 1)
        
        # Sibling of child_on_X_path. This is the root of the subtree where v might be.
        # child_on_X_path must be >= 2 (since LCA >= 1).
        # So, child_on_X_path ^ 1 correctly gives its sibling.
        sibling_branch_root = child_on_X_path ^ 1
        
        k_down = K - k_up # dist(LCA, v)
        # v is in subtree of sibling_branch_root.
        # dist(LCA, sibling_branch_root) is 1.
        # So, dist(sibling_branch_root, v) must be k_down - 1.
        # Since k_up <= K-1, k_down = K - k_up >= 1. So k_down - 1 >= 0.
        # The relative depth for get_nodes_at_level is non-negative.
        
        ans += get_nodes_at_level(sibling_branch_root, k_down - 1, N)
        
    sys.stdout.write(str(ans) + "
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()