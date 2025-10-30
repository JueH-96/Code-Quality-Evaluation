import sys

MOD = 998244353

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    
    special_edges = []
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        special_edges.append((u, v))
    
    # 1. Identify all special vertices: X_i, Y_i, and starting vertex 1
    V_S_set = set()
    V_S_set.add(1)
    for u, v in special_edges:
        V_S_set.add(u)
        V_S_set.add(v)
    
    V_S = sorted(list(V_S_set))
    P = len(V_S) # Number of unique special vertices
    
    # Map vertex value to its index in V_S
    idx_map = {v: i for i, v in enumerate(V_S)}
    
    # 2. Precompute prev_s and dist_to_prev_s
    # prev_s[v] is the first vertex in V_S found when going backwards from v along cyclic edges.
    # dist_to_prev_s[v] is the length of path from prev_s[v] to v.
    is_special = [False] * (N + 1)
    for v in V_S:
        is_special[v] = True
    
    prev_s = [0] * (N + 1)
    dist_to_prev_s = [0] * (N + 1)
    
    # Calculate for each `v` from `1` to `N` based on its cyclic predecessor
    for i in range(1, N + 1):
        if is_special[i]:
            prev_s[i] = i
            dist_to_prev_s[i] = 0
        else:
            # Cyclic predecessor: (i-1) if i>1, N if i=1
            pred_node = (i - 2 + N) % N + 1 # Converts 1-based index correctly
            prev_s[i] = prev_s[pred_node]
            dist_to_prev_s[i] = dist_to_prev_s[pred_node] + 1

    # 3. Precompute incoming special edges
    incoming_special_edges_to = [[] for _ in range(N + 1)]
    for x, y in special_edges:
        incoming_special_edges_to[y].append(x)
    
    # 4. Dynamic Programming
    # dp_vals[k][idx] stores f[k][V_S[idx]]
    dp_vals = [[0] * P for _ in range(K + 1)]
    # total_counts[k] stores sum(f[k][u] for u in 1..N)
    total_counts = [0] * (K + 1)

    # Base case
    dp_vals[0][idx_map[1]] = 1
    total_counts[0] = 1

    for k in range(1, K + 1):
        # Calculate dp_vals[k][i] for each special vertex V_S[i]
        for i in range(P): 
            u = V_S[i] # Current target vertex in V_S
            
            # Contribution from special edges
            # Sum f[k-1][v_from] for all v_from -> u special edges
            for v_from in incoming_special_edges_to[u]:
                if v_from in V_S_set: # Only consider v_from if it's a special vertex itself
                    dp_vals[k][i] = (dp_vals[k][i] + dp_vals[k-1][idx_map[v_from]]) % MOD
            
            # Contribution from cyclic path
            # This path is `prev_s[pred_u] -> ... -> pred_u -> u`
            pred_u = (u - 2 + N) % N + 1
            source_s_val = prev_s[pred_u] # The special vertex where this cyclic path started
            cyclic_path_len = dist_to_prev_s[pred_u] + 1 # Length of this cyclic path segment
            
            if k >= cyclic_path_len:
                dp_vals[k][i] = (dp_vals[k][i] + dp_vals[k - cyclic_path_len][idx_map[source_s_val]]) % MOD
        
        # Calculate total_counts[k]
        # total_counts[k] = total_counts[k-1] + sum_{j=1..M} f[k-1][X_j]
        total_counts[k] = total_counts[k-1]
        for x_j, _ in special_edges: # iterate through all X_j vertices
            total_counts[k] = (total_counts[k] + dp_vals[k-1][idx_map[x_j]]) % MOD

    sys.stdout.write(str(total_counts[K]) + "
")

solve()