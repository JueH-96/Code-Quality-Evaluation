import sys

# Increase recursion limit for deep DFS paths
# Max N is 2*10^5. A path graph would hit recursion depth N.
sys.setrecursionlimit(2 * 10**5 + 5000) 

def solve():
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N)]

    if N == 0: # Should not happen based on constraints 1 <= N
        print(-1)
        return
    
    # Read N-1 edges for a tree. If N=1, this loop range is 0.
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # 0-indexed
        v -= 1 # 0-indexed
        adj[u].append(v)
        adj[v].append(u)

    global_max_alkane_size = -1
    
    # dp0_u = (count, has_deg4_node_in_fragment):
    #   u connects to parent, u's degree in the fragment is 1.
    # dp1_u = (count, has_deg4_node_in_fragment):
    #   u connects to parent, u's degree in the fragment is 4.
    
    def dfs(u, p):
        nonlocal global_max_alkane_size

        # Base state for dp0_u: u connects upwards as degree 1.
        # Fragment is just u itself. count = 1. No deg4 node from u.
        dp0_u = (1, False)

        # Base state for dp1_u: u connects upwards as degree 4.
        # Requires 3 children connections. Initialize as impossible.
        dp1_u = (-1, False) 

        # child_contributions stores (value, has_deg4_flag) pairs from children of u
        # that connect to u to form part of u's fragment.
        child_contributions = []
        
        for v in adj[u]:
            if v == p: # Don't go back to parent
                continue
            
            # Recursively get results for child v
            dp0_v, dp1_v = dfs(v, u)

            # Option 1: v connects to u, acting as degree 1 in its own fragment part.
            v_val1, v_has_deg4_1 = dp0_v
            
            # Option 2: v connects to u, acting as degree 4 in its own fragment part.
            v_val2, _ = dp1_v # The second element of dp1_v is True if v_val2 != -1
            v_has_deg4_2 = (v_val2 != -1) # True if v forms a valid deg4 structure

            best_v_val = -1
            best_v_has_deg4 = False

            # Determine best way for v to contribute to u's fragment
            if v_val1 != -1: # If v can connect as degree 1
                best_v_val = v_val1
                best_v_has_deg4 = v_has_deg4_1
            
            if v_val2 != -1: # If v can connect as degree 4
                if v_val2 > best_v_val:
                    best_v_val = v_val2
                    best_v_has_deg4 = True # v itself is degree 4
                elif v_val2 == best_v_val: # Same value, prioritize if it brings a deg4 node
                    best_v_has_deg4 = best_v_has_deg4 or True # True if either option had deg4
            
            if best_v_val != -1: # If v can form a valid fragment to attach to u
                child_contributions.append((best_v_val, best_v_has_deg4))
        
        # Sort children by their contribution value (descending).
        # If values are equal, prioritize ones with has_deg4=True.
        child_contributions.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Calculate dp1_u: u connects upwards as degree 4.
        # Needs 1 edge to parent (implicit) and 3 edges to children.
        if len(child_contributions) >= 3:
            current_val = 1 # For u itself
            # No need to track has_deg4 from children; u itself is degree 4.
            for i in range(3): # Sum contributions from top 3 children
                current_val += child_contributions[i][0]
            dp1_u = (current_val, True) # u is degree 4, so fragment has a deg4 node

        # Consider alkanes rooted at u (u does not connect to its parent p).
        # These update global_max_alkane_size.
        
        # Case 1: Alkane rooted at u, u has degree 1 in this alkane.
        # u connects to 1 child. This child's branch MUST provide a degree 4 node.
        if len(child_contributions) >= 1:
            # Iterate through all child contributions
            for c_val, c_has_d4 in child_contributions:
                if c_has_d4: # If this child's branch has a degree 4 node
                    current_size = 1 + c_val # 1 for u
                    if global_max_alkane_size == -1 or current_size > global_max_alkane_size:
                        global_max_alkane_size = current_size
        
        # Case 2: Alkane rooted at u, u has degree 4 in this alkane.
        # u connects to 4 children. u itself provides the degree 4 node.
        if len(child_contributions) >= 4:
            current_val = 1 # For u itself
            for i in range(4): # Sum contributions from top 4 children
                current_val += child_contributions[i][0]
            
            # This forms a valid alkane because u is degree 4.
            if global_max_alkane_size == -1 or current_val > global_max_alkane_size:
                global_max_alkane_size = current_val
        
        return dp0_u, dp1_u

    if N > 0: 
        dfs(0, -1) # Start DFS from node 0, with a dummy parent -1
    # If N=0 or N=1, dfs might not run or runs on a single node.
    # For N=1: dfs(0,-1) -> child_contributions empty.
    # dp0_0=(1,F), dp1_0=(-1,F). global_max_alkane_size remains -1. Correct.
    # Smallest alkane (K_1,4) has 5 vertices. If N < 5, result is typically -1 unless
    # specific structures are possible (which they aren't for alkanes).
    # The DP correctly handles this; no explicit N < 5 check is strictly needed.

    print(global_max_alkane_size)

solve()