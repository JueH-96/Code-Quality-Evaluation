import sys

# Increase recursion limit to handle deep trees (N up to 10^5)
# Default is often 1000 or 3000, which is insufficient for N=10^5 in a path graph.
sys.setrecursionlimit(2 * 10**5)

def solve():
    N = int(sys.stdin.readline())

    # Adjacency list for the tree. Using 1-indexed nodes (1 to N).
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Read C values. C_val[i] will store C_i (for 1-indexed node i).
    C_raw = list(map(int, sys.stdin.readline().split()))
    C_val = [0] * (N + 1)
    for i in range(N):
        C_val[i+1] = C_raw[i]

    # subtree_C_sum[u]: Sum of C_j for all j in the subtree rooted at u (including u itself).
    subtree_C_sum = [0] * (N + 1)
    
    # subtree_f_sum[u]: Sum of C_j * d(u, j) for all j in the subtree rooted at u.
    # This is a partial sum related to f(u), considering only nodes within the subtree.
    subtree_f_sum = [0] * (N + 1)
    
    # dp[u]: Stores the final f(u) value for each node u.
    dp = [0] * (N + 1)
    
    # total_sum_C: Sum of all C_i values across the entire tree.
    total_sum_C = 0

    # DFS1: A post-order (bottom-up) traversal to calculate subtree_C_sum and subtree_f_sum.
    # This function calculates sums for the current node 'u' based on its children's sums.
    def dfs1(u, p):
        # Initialize with the current node's C value.
        subtree_C_sum[u] = C_val[u]
        # Initialize subtree_f_sum for u. The distance d(u,u) is 0, so no initial contribution from u itself.
        subtree_f_sum[u] = 0 
        
        for v in adj[u]:
            if v == p: # Skip parent to avoid going back up and infinite recursion
                continue
            
            dfs1(v, u) # Recurse for child v
            
            # Aggregate sums from children.
            # subtree_C_sum[u] accumulates C values from its subtree.
            subtree_C_sum[u] += subtree_C_sum[v]
            
            # subtree_f_sum[u] accumulates f values from its subtree.
            # For any node 'j' in child 'v's subtree:
            # d(u, j) = d(v, j) + 1.
            # So, sum(C_j * d(u, j)) for j in v's subtree
            # = sum(C_j * (d(v, j) + 1))
            # = sum(C_j * d(v, j)) + sum(C_j)
            # This is subtree_f_sum[v] + subtree_C_sum[v].
            subtree_f_sum[u] += subtree_f_sum[v] + subtree_C_sum[v]

    # Start the first DFS from node 1 (arbitrary root). Parent is 0 (dummy).
    dfs1(1, 0)

    # After DFS1 completes, subtree_C_sum[1] will contain the sum of all C_i values across the tree.
    total_sum_C = subtree_C_sum[1]
    
    # And subtree_f_sum[1] will be the full f(1) value, as 1 is the root and all other nodes are in its subtree.
    # f(1) = sum(C_i * d(1, i)) which is exactly what subtree_f_sum[1] calculated.
    dp[1] = subtree_f_sum[1]

    # Initialize minimum f value found so far to positive infinity.
    min_f_val = float('inf')

    # DFS2: A pre-order (top-down) traversal to calculate dp[u] for all nodes using re-rooting.
    def dfs2(u, p):
        nonlocal min_f_val # Declare min_f_val as nonlocal to modify the outer scope variable
        min_f_val = min(min_f_val, dp[u]) # Update global minimum with current node's f value

        for v in adj[u]:
            if v == p: # Skip parent to avoid redundant calculation and going backward
                continue
            
            # Calculate dp[v] based on dp[u] using the re-rooting formula:
            # dp[v] = dp[u] - 2 * subtree_C_sum[v] + total_sum_C
            # Explanation:
            # When moving the root from u to its child v:
            # - For nodes 'j' in v's subtree, d(v, j) = d(u, j) - 1. Their contribution to f decreases by C_j.
            #   Total decrease for v's subtree: -sum(C_j for j in subtree v) = -subtree_C_sum[v]
            # - For nodes 'j' NOT in v's subtree (i.e., 'u' and all nodes on the other side of edge (u,v)), 
            #   d(v, j) = d(u, j) + 1. Their contribution to f increases by C_j.
            #   Total increase for nodes outside v's subtree: +sum(C_j for j not in subtree v)
            #   = +(total_sum_C - subtree_C_sum[v])
            # The net change from dp[u] to dp[v] is:
            # (-subtree_C_sum[v]) + (total_sum_C - subtree_C_sum[v]) = total_sum_C - 2 * subtree_C_sum[v]
            dp[v] = dp[u] + total_sum_C - 2 * subtree_C_sum[v]
            
            dfs2(v, u) # Recurse for child v

    # Start the second DFS from the same root (node 1).
    dfs2(1, 0)

    # Print the minimum f value found.
    sys.stdout.write(str(min_f_val) + '
')

# Execute the main solve function
solve()