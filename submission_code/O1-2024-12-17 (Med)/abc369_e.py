def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # We will parse the input_data manually for speed in Python
    # (since large input can be slow with repeated input() calls).
    # Format reminder:
    # N M
    # U_1 V_1 T_1
    # ...
    # U_M V_M T_M
    # Q
    # K_1
    # B_{1,1} ... B_{1,K_1}
    # ...
    # K_Q
    # B_{Q,1} ... B_{Q,K_Q}

    INF = 10**15  # A sufficiently large number (bigger than any possible path sum).
    
    # --- 1) Read N, M ---
    ptr = 0
    N = int(input_data[ptr]); ptr += 1
    M = int(input_data[ptr]); ptr += 1
    
    # --- 2) Build adjacency matrix for Floyd-Warshall ---
    dist = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    
    edges = [None]*M  # we also store each edge (u,v,t) in 0-based for queries
    
    # --- 3) Read the M edges ---
    for i in range(M):
        u  = int(input_data[ptr]); ptr += 1
        v  = int(input_data[ptr]); ptr += 1
        t  = int(input_data[ptr]); ptr += 1
        u -= 1
        v -= 1
        # Update adjacency matrix with the minimum cost if multiple edges
        if t < dist[u][v]:
            dist[u][v] = t
            dist[v][u] = t
        edges[i] = (u, v, t)
    
    # --- 4) Floyd-Warshall to get all-pairs shortest path in dist ---
    #    O(N^3) = 400^3 = 64e6 in worst case, borderline but can pass in optimized Python/PyPy.
    for k in range(N):
        d_k = dist[k]  # for speed, avoid repeated dist[k] lookups
        for i in range(N):
            d_i_k = dist[i][k]  # dist[i][k]
            d_i = dist[i]
            # unroll the "for j in range(N):" carefully
            for j in range(N):
                alt = d_i_k + d_k[j]
                if d_i[j] > alt:
                    d_i[j] = alt
    
    # --- 5) Now handle the Q queries. ---
    Q = int(input_data[ptr]); ptr += 1

    # Pre-allocate some arrays to speed up repeated DP usage
    # We will do a TSP-DP for each query with up to K=5 edges.
    # The DP array: dp[mask][last], mask up to 2^5=32, last up to 5 -> 160 elements.
    # We'll reuse a 2D list for DP (to limit memory allocations).
    DP = [[INF]*5 for _ in range(1<<5)]
    
    import math
    import itertools
    
    out = []
    for _q in range(Q):
        K = int(input_data[ptr]); ptr += 1
        # read forced bridges indices
        forced_inds = [int(input_data[ptr + i]) - 1 for i in range(K)]
        ptr += K
        forced_edges = [edges[idx] for idx in forced_inds]  # (u,v,t) in 0-based
        
        if K == 0:
            # If it (somehow) happened, just dist(1->N)
            # (Problem statement says K_i>=1, so unlikely, but just in case.)
            out.append(str(dist[0][N-1]))
            continue
        
        # We will do the "direction-assignment + TSP DP" approach.
        # For each of the K forced edges, we can traverse it from u->v or v->u.
        # The crossing time T is the same either way, but the "start" and "end" node differ.
        # We then run a TSP over the forced edges to find the minimal route that
        # starts at node 1 (index 0), uses all forced edges exactly once in some order,
        # and then ends at node N (index N-1).
        
        best_answer = INF
        
        # Pre-extract (u, v, t) in arrays for speed
        us = [e[0] for e in forced_edges]
        vs = [e[1] for e in forced_edges]
        ts = [e[2] for e in forced_edges]
        
        # There are 2^K possible direction assignments
        # direction bit i says 0 => use (u_i->v_i), 1 => use (v_i->u_i).
        # We'll run a TSP DP for each assignment.
        # TSP DP has 2^K*K states, each transition enumerates the next forced edge to use.
        
        for directions in range(1 << K):
            # Build arrays starts[i], ends[i], costS[i], costE[i], costC[i][j].
            # starts[i] = the node where forced edge i starts in the chosen direction
            # ends[i]   = the node where forced edge i ends   in the chosen direction
            # costS[i]  = cost(1-> starts[i]) + ts[i]
            # costE[i]  = cost(ends[i]-> N)
            # costC[i][j] = cost(ends[i] -> starts[j]) + ts[j]
            
            starts = [0]*K
            ends   = [0]*K
            for i in range(K):
                d_i = (directions >> i) & 1
                if d_i == 0:
                    # u->v
                    starts[i] = us[i]
                    ends[i]   = vs[i]
                else:
                    # v->u
                    starts[i] = vs[i]
                    ends[i]   = us[i]
            
            costS = [0]*K
            costE = [0]*K
            costC = [[0]*K for _ in range(K)]
            
            for i in range(K):
                costS[i] = dist[0][starts[i]] + ts[i]   # from 1(=0) to start[i], cross the edge i
                costE[i] = dist[ends[i]][N-1]          # from ends[i] to N(=N-1)
            for i in range(K):
                ei = ends[i]
                row = costC[i]
                for j in range(K):
                    sj = starts[j]
                    row[j] = dist[ei][sj] + ts[j]  # travel from ends[i] to starts[j], then cross edge j
            
            # Now do the TSP DP over edges 0..K-1
            # dp[mask][i] = min cost to have used edges in mask with i as last used
            # We'll re-init DP array for each direction assignment.
            for m in range(1<<K):
                dpm = DP[m]
                for _i in range(K):
                    dpm[_i] = INF
            
            # base case
            for i in range(K):
                DP[1 << i][i] = costS[i]
            
            # transitions
            for mask in range(1<<K):
                dpm = DP[mask]
                for i in range(K):
                    base_cost = dpm[i]
                    if base_cost == INF:
                        continue
                    # try next edge j not in mask
                    for j in range(K):
                        if not (mask & (1 << j)):
                            new_mask = mask | (1 << j)
                            new_cost = base_cost + costC[i][j]
                            if new_cost < DP[new_mask][j]:
                                DP[new_mask][j] = new_cost
            
            full_mask = (1 << K) - 1
            # add cost to reach N
            candidate = INF
            dpf = DP[full_mask]
            for i in range(K):
                val = dpf[i]
                if val == INF: 
                    continue
                cend = val + costE[i]
                if cend < candidate:
                    candidate = cend
            
            if candidate < best_answer:
                best_answer = candidate
        
        out.append(str(best_answer))
    
    print("
".join(out))

# Do not forget to call main()!
if __name__ == "__main__":
    main()