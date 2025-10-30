import sys

# Increase recursion limit for deep trees
# NK can be up to 2 * 10^5. Max depth can be NK.
sys.setrecursionlimit(2 * 10**5 + 50) 

N_global = 0
K_global = 0
adj_global = []
count_paths_global = 0

# dfs(u, p) returns:
# - L (where 1 <= L < K_global): u is an endpoint of a path segment of length L (u, child, ...).
#                                This segment needs to be extended by u's parent p.
# - 0: u and its subtree are "balanced".
# - -1: impossible to decompose.

def dfs(u, p):
    global count_paths_global

    # List of path segments starting at u, going via a child c, and having length < K_global.
    # Their lengths are (dfs(c,u) + 1).
    child_segments_lt_K = [] 
    
    for v_node in adj_global[u]:
        if v_node == p:
            continue
        
        res_child = dfs(v_node, u) # This is the length of path v_node... passed up by v_node
        
        if res_child == -1: # Impossible in child's subtree
            return -1
        
        if res_child == 0: # Child v_node's subtree is balanced, edge (u, v_node) is effectively "cut".
            continue
        
        # res_child > 0 means path v_node... has length res_child (which is < K_global).
        # Extended via u: u-v_node... The new length is res_child + 1.
        current_segment_length = res_child + 1
        
        if current_segment_length < K_global:
            child_segments_lt_K.append(current_segment_length)
        elif current_segment_length == K_global:
            count_paths_global += 1 # Path u-v_node... is a K_global-path.
        else: 
            # current_segment_length > K_global. This implies res_child + 1 > K_global.
            # Since res_child < K_global, res_child can be at most K_global-1.
            # So res_child + 1 can be at most K_global.
            # Thus, this > K_global case should not be reached. For safety, treat as error.
            return -1 
            
    # Sort for determinism if needed, though logic here doesn't rely on specific order for multiple <K segments.
    # child_segments_lt_K.sort(reverse=True) 

    # Determine what u returns to its parent p.
    # u can either start a new path of length 1 (just u) upwards,
    # or extend one of the child_segments_lt_K upwards.
    # All other segments must have been resolved (i.e., they became K_global-paths and were counted).
    
    if p == 0: # u is the root node
        if not child_segments_lt_K:
            # No segments < K_global from children. The only remaining part is u itself.
            # This forms a path segment of length 1 (just u).
            # This path must be a K_global-path.
            if K_global == 1: # Path <u> has length 1. If K_global=1, this is a K_global-path.
                count_paths_global += 1
                return 0 # Root is balanced
            else: # Path <u> has length 1, but K_global > 1. Stranded.
                return -1
        elif len(child_segments_lt_K) == 1:
            # One segment u-v_node... of length L = child_segments_lt_K[0] where L < K_global.
            # This segment must itself be a K_global-path.
            # (This implies L must actually be K_global, contradicting L < K_global, unless N_global=1 case, etc.)
            # The problem asks for K_global length paths. If this path from root is L < K_global, it's not valid.
            # The only way it could be valid is if L is actually K_global.
            # But child_segments_lt_K stores lengths strictly less than K_global.
            # So this path is stranded.
            return -1
        else: # len(child_segments_lt_K) > 1
              # Multiple segments < K_global starting at root (e.g., u-v1... and u-v2...).
              # Since u can only be part of one path, this is impossible.
            return -1
    else: # u is not the root node
        if not child_segments_lt_K:
            # No segments < K_global from children were passed to u to extend.
            # u itself can form a path segment of length 1 to be passed upwards to p.
            return 1
        elif len(child_segments_lt_K) == 1:
            # One segment u-v_node... of length L = child_segments_lt_K[0] (where L < K_global).
            # This segment can be passed upwards to p.
            return child_segments_lt_K[0]
        else: # len(child_segments_lt_K) > 1
              # Multiple segments < K_global start at u.
              # u can only be an endpoint for one path going upwards (either as u...p or c...u...p).
              # If it passes one segment u-v1... upwards, other segments like u-v2... are stranded. Impossible.
            return -1

def solve():
    global N_global, K_global, adj_global, count_paths_global
    
    N_val, K_val = map(int, sys.stdin.readline().split())
    N_global = N_val
    K_global = K_val

    num_total_vertices = N_global * K_global
    
    if K_global == 1:
        print("Yes")
        return

    # Constraints: N, K >= 1, so NK >= 1.
    # If NK=1, then N=1, K=1. Covered by K_global==1 case.
    # So num_total_vertices >= 1.
    
    adj_global = [[] for _ in range(num_total_vertices + 1)]
    
    # Number of edges is num_total_vertices - 1.
    # If num_total_vertices = 1, this loop range is 0, no edges read. Correct.
    for _ in range(num_total_vertices - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj_global[u].append(v)
        adj_global[v].append(u)

    count_paths_global = 0
    
    # Pick vertex 1 as root. Parent of root is 0 (sentinel).
    # Vertices are 1 to NK. If NK>=1, vertex 1 exists.
    root_node = 1 
    
    # In case NK=0 (not possible by constraints but defensive)
    if num_total_vertices == 0:
         # N=0 or K=0. If N=0, need 0 paths. Vacuously true?
         # Problem constraints say N, K >= 1.
         print("No") # Or based on interpretation for N=0.
         return


    root_dfs_res = dfs(root_node, 0) 

    if root_dfs_res == 0 and count_paths_global == N_global:
        print("Yes")
    else:
        print("No")

solve()