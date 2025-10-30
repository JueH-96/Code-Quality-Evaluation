import sys

sys.setrecursionlimit(200005)

def solve():
    N = int(sys.stdin.readline())
    
    if N == 0:
        print(-1)
        return
        
    adj = [[] for _ in range(N + 1)]
    degrees = [0] * (N + 1)
    if N > 1:
        for _ in range(N - 1):
            u, v = map(int, sys.stdin.readline().split())
            adj[u].append(v)
            adj[v].append(u)
            degrees[u] += 1
            degrees[v] += 1

    max_degree_T = max(degrees) if N > 0 else 0

    # If no node in T has degree >= 4, no induced subgraph can have a node with degree >= 4.
    # Alkane requires at least one degree 4 vertex.
    # Special case: N=1, max_degree=0. No alkane.
    # For N>1, max_degree < 4 implies no alkane.
    if max_degree_T < 4 and N > 1:
        print(-1)
        return
        
    # For N=1, max_degree is 0. Cannot form alkane. Answer -1.
    # This will be handled by the DP returning INF.

    INF = -int(1e9) # Use a large negative number for impossible states

    # dfs(u, parent) returns (max_closed_size_in_subtree, dp_u_map)
    # max_closed_size_in_subtree[h]: max size of an alkane completely within subtree(u), requires deg4 if h=1
    # dp_u_map: map<int, list<int>> where map[k] = [size_h0, size_h1]
    # k: number of children of u connected in the partial structure rooted at u
    # size_h0: max size if no deg4 vertex found yet *within the partial structure rooted at u excluding u's potential deg4*
    # size_h1: max size if at least one deg4 vertex found *within the partial structure rooted at u excluding u's potential deg4*
    # The returned map keys k are 0 to (number of children of u).

    def dfs(u, p):
        # max_closed_size_u[0]: no deg4 in this structure
        # max_closed_size_u[1]: has deg4 in this structure
        # Initial value INF for requiring deg4, as an empty graph (size 0) doesn't have deg4.
        max_closed_size_u = [INF, INF] 

        # dp_u_map: {num_connected_children : [size_h0, size_h1]}
        # Initial state: u included, connected to 0 children processed, size 1, no deg4 yet.
        # Map key 0 means 0 children connected *out of the total children of u*.
        current_dp_u = {0: [1, INF]}

        children = [v for v in adj[u] if v != p]

        for v in children:
            max_closed_size_v, dp_v_map = dfs(v, u)

            # Update max_closed_size_u from closed subgraphs in child subtree
            max_closed_size_u[0] = max(max_closed_size_u[0], max_closed_size_v[0])
            max_closed_size_u[1] = max(max_closed_size_u[1], max_closed_size_v[1])

            # Combine current_dp_u with results from dp_v_map (knapsack-like)
            new_tmp_dp = {}

            # Function to update map entry
            def update_dp(dp_map, key, h, value):
                if key not in dp_map:
                    dp_map[key] = [INF, INF]
                dp_map[key][h] = max(dp_map[key][h], value)

            # Get states from child v needed for combining
            # Option 1 (no conn to u): v needs 0 upward connection.
            # This means v's degree within its partial structure must be 1 or 4.
            # Relevant dp_v states: dp_v_map[1][h] and dp_v_map[4][h]. Also closed_v[h].
            
            size_v_no_conn_h = [max_closed_size_v[0], max_closed_size_v[1]] # Default from closed

            # Check if dp_v_map has key 1 (v connected to 1 child)
            if 1 in dp_v_map:
                 if dp_v_map[1][0] != INF: size_v_no_conn_h[0] = max(size_v_no_conn_h[0], dp_v_map[1][0])
                 if dp_v_map[1][1] != INF: size_v_no_conn_h[1] = max(size_v_no_conn_h[1], dp_v_map[1][1])
            
            # Check if dp_v_map has key 4 (v connected to 4 children)
            if 4 in dp_v_map:
                 if dp_v_map[4][0] != INF: size_v_no_conn_h[0] = max(size_v_no_conn_h[0], dp_v_map[4][0])
                 if dp_v_map[4][1] != INF: size_v_no_conn_h[1] = max(size_v_no_conn_h[1], dp_v_map[4][1])


            # Option 2 (conn to u): v needs 1 upward connection.
            # This means v's degree within its partial structure + 1 must be 1 or 4.
            # v connected to k_v children + 1 upward edge. So k_v + 1 = 1 or 4.
            # k_v must be 0 or 3.
            # Relevant dp_v states: dp_v_map[0][h] and dp_v_map[3][h].
            
            size_v_conn_h_k0 = [INF, INF]
            if 0 in dp_v_map: # k_v = 0
                 size_v_conn_h_k0 = list(dp_v_map[0]) # Make a copy

            size_v_conn_h_k3 = [INF, INF]
            if 3 in dp_v_map: # k_v = 3
                 size_v_conn_h_k3 = list(dp_v_map[3]) # Make a copy


            # Iterate through current_dp_u states
            for j_u_prev, sizes_u_prev in current_dp_u.items():
                for h_u_prev in range(2):
                    if sizes_u_prev[h_u_prev] == INF: continue

                    # Option 1: Don't connect u to v
                    if size_v_no_conn_h[0] != INF:
                        update_dp(new_tmp_dp, j_u_prev, h_u_prev | 0, sizes_u_prev[h_u_prev] + size_v_no_conn_h[0])
                    if size_v_no_conn_h[1] != INF:
                        update_dp(new_tmp_dp, j_u_prev, h_u_prev | 1, sizes_u_prev[h_u_prev] + size_v_no_conn_h[1])

                    # Option 2: Connect u to v
                    # New child connection count at u is j_u_prev + 1
                    j_u_curr = j_u_prev + 1
                    
                    # v uses k_v = 0
                    if size_v_conn_h_k0[0] != INF:
                        update_dp(new_tmp_dp, j_u_curr, h_u_prev | 0, sizes_u_prev[h_u_prev] + size_v_conn_h_k0[0])
                    if size_v_conn_h_k0[1] != INF:
                        update_dp(new_tmp_dp, j_u_curr, h_u_prev | 1, sizes_u_prev[h_u_prev] + size_v_conn_h_k0[1])

                    # v uses k_v = 3
                    if size_v_conn_h_k3[0] != INF:
                        update_dp(new_tmp_dp, j_u_curr, h_u_prev | 0, sizes_u_prev[h_u_prev] + size_v_conn_h_k3[0])
                    if size_v_conn_h_k3[1] != INF:
                        update_dp(new_tmp_dp, j_u_curr, h_u_prev | 1, sizes_u_prev[h_u_prev] + size_v_conn_h_k3[1])

            current_dp_u = new_tmp_dp

        # After processing all children
        # Update max_closed_size_u using final current_dp_u states
        # A structure rooted at u with j children connections becomes a closed alkane
        # if its degree j is 1 or 4.
        # Degree of u in the potential subgraph is j (number of children connected).
        
        final_dp_u = current_dp_u

        for j, sizes in final_dp_u.items():
            # If u's degree in the subgraph is 1
            if j == 1:
                # Needs h=1 (deg4 exists elsewhere in the partial structure)
                if sizes[1] != INF:
                     max_closed_size_u[1] = max(max_closed_size_u[1], sizes[1])
            # If u's degree in the subgraph is 4
            elif j == 4:
                # Deg4 exists (u is deg4). Can be h=0 or h=1 from children.
                if sizes[0] != INF:
                    max_closed_size_u[1] = max(max_closed_size_u[1], sizes[0])
                if sizes[1] != INF:
                    max_closed_size_u[1] = max(max_closed_size_u[1], sizes[1])

        # The dp_u_map returned is the final_dp_u map.
        # Its keys represent number of children connected (0 to num_children_u).
        return (max_closed_size_u, final_dp_u)

    # Find a root (any node is fine, let's use 1)
    root = 1
    final_closed, _ = dfs(root, -1)

    result = final_closed[1] # Max size of alkane needing deg4

    # Alkane size must be >= 5 (min one deg4 + 4 leaves)
    # A single node (size 1, deg 0), or a path A-B (size 2, deg 1,1) cannot be an alkane.
    # The DP naturally produces valid sizes or INF.
    # If the max size found needing deg4 is INF, it's impossible.
    if result == INF:
        print(-1)
    else:
        print(result)

solve()