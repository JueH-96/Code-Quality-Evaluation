import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    
    shortcuts_input = []
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        shortcuts_input.append((u, v))

    MOD = 998244353

    # ans stores the total number of ways.
    # Initialize with 1 way for the path using 0 shortcut edges (if K >= 0).
    # Path: 1 -> 1+1 -> ... for K steps. This path is valid if K >= 0.
    # L_core for this path is 0. Remaining K-0 steps must be non-negative.
    ans = 1
    # Problem constraints: K >= 1. If K=0 was allowed, ans=1 (stay at vertex 1).

    # dp_curr[edge_idx] maps {length_L_core: count_of_paths}
    # Stores paths ending with shortcut 'edge_idx' having used (current_s_count - 1) shortcuts.
    dp_curr = [{} for _ in range(M)]

    # Initialize for s_count = 1 (paths with 1 shortcut edge)
    for j in range(M):
        X_j, Y_j = shortcuts_input[j]
        
        # Length from vertex 1 to X_j using only cycle edges:
        # (X_j - 1) if 1-indexed vertices. (X_j - 1 + N) % N accounts for wrap-around.
        dist_1_to_Xj = (X_j - 1 + N) % N
        
        l_core = dist_1_to_Xj + 1 # +1 for the shortcut X_j -> Y_j
        
        if l_core <= K:
            # dp_curr[j] stores paths with 1 shortcut, ending in edge j.
            # count = 1 for this specific sequence (just edge j)
            dp_curr[j][l_core] = dp_curr[j].get(l_core, 0) + 1
            ans = (ans + 1) % MOD # Add this 1 way to total.
            
    # Iterate for s_count = 2 up to K (max number of shortcuts in a path of length K is K)
    # L_core >= s_count, so s_count <= L_core <= K. Thus, s_count loop up to K.
    for s_count in range(2, K + 1):
        dp_next = [{} for _ in range(M)]
        processed_any_path = False # Flag to check if any valid path was extended

        for p_idx in range(M): # Previous shortcut was p_idx
            # Y_p is the vertex where the (s_count-1)-th shortcut (p_idx) ended.
            _unused_X_p, Y_p = shortcuts_input[p_idx]

            for l_p, count_p in dp_curr[p_idx].items():
                # l_p is L_core of a path with (s_count-1) shortcuts, ending at Y_p via p_idx.
                # count_p is the number of such sequences of (s_count-1) shortcuts.
                
                for j_idx in range(M): # Current shortcut is j_idx
                    X_j, _unused_Y_j = shortcuts_input[j_idx]
                    
                    # Minimal cycle edges from Y_p to X_j
                    dist_Yp_to_Xj = (X_j - Y_p + N) % N
                    
                    l_new = l_p + dist_Yp_to_Xj + 1 # +1 for shortcut j_idx
                    
                    if l_new <= K:
                        dp_next[j_idx][l_new] = (dp_next[j_idx].get(l_new, 0) + count_p) % MOD
                        ans = (ans + count_p) % MOD
                        processed_any_path = True
        
        dp_curr = dp_next
        if not processed_any_path:
            # If no paths were extended to s_count shortcuts with L_core <= K,
            # then no paths with >s_count shortcuts can be formed either.
            break
            
    print(ans)

solve()