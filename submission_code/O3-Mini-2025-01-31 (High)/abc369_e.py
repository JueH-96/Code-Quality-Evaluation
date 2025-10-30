# YOUR CODE HERE
def main():
    import sys,math
    import numpy as np
    # Read the entire input (using sys.stdin.buffer for speed)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # We choose an INF that is much larger than any possible path.
    # Note: T_i up to 10^9, worst-case path may be a few hundred such edges.
    INF = 10**15
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Build the NxN distance matrix; make sure to use np.int64.
    d = np.full((N, N), INF, dtype=np.int64)
    for i in range(N):
        d[i, i] = 0

    # Store the M bridges (each: (u,v,t) in 0-indexed form)
    bridges = [None] * M
    for i in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        t = int(next(it))
        bridges[i] = (u, v, t)
        # Update with the minimum cost if several bridges connect the same pair.
        if t < d[u, v]:
            d[u, v] = t
            d[v, u] = t
        else:
            if t < d[v, u]:
                d[v, u] = t
                d[u, v] = t

    # Compute all-pairs shortest paths using a vectorized Floyd Warshall.
    for k in range(N):
        d = np.minimum(d, d[:, k:k+1] + d[k:k+1, :])
    # Convert the numpy matrix to a Python list-of-lists for fast inner–loop indexing.
    dist = d.tolist()
    
    # Process Q queries.
    Q = int(next(it))
    out_lines = []
    for _ in range(Q):
        k_val = int(next(it))
        forced_edges = []
        for j in range(k_val):
            bid = int(next(it))
            forced_edges.append(bridges[bid-1])
        # For each forced edge (u,v,t) consider both orientations:
        # Option 0: "start" at u (finish at v) and Option 1: "start" at v (finish at u)
        forced_opts = []
        for (u, v, t) in forced_edges:
            forced_opts.append( ((u, v), (v, u)) )
        # dp[mask][i][o] = best cost when the set 'mask' of forced edges is used,
        # with the last used forced edge being index i in orientation o.
        size = 1 << k_val
        dp = [ [ [INF, INF] for _ in range(k_val) ] for _ in range(size) ]
        # For each forced edge, the initial cost is:
        # cost = free-distance from island 1 (index 0) to the "start" node + t.
        for i in range(k_val):
            for o in (0, 1):
                start_node = forced_opts[i][o][0]
                cost_here = dist[0][start_node] + forced_edges[i][2]
                dp[1 << i][i][o] = cost_here
        # Now do the DP over all subsets (bitmask) of forced edges.
        for mask in range(size):
            for i in range(k_val):
                if mask & (1 << i):
                    for o in (0, 1):
                        cur_cost = dp[mask][i][o]
                        if cur_cost >= INF:
                            continue
                        # current endpoint (where we “landed” after using forced edge i in orientation o)
                        cur_end = forced_opts[i][o][1]
                        for j in range(k_val):
                            if mask & (1 << j):
                                continue
                            for r in (0, 1):
                                start_j = forced_opts[j][r][0]
                                cand = cur_cost + dist[cur_end][start_j] + forced_edges[j][2]
                                new_mask = mask | (1 << j)
                                if cand < dp[new_mask][j][r]:
                                    dp[new_mask][j][r] = cand
        best = INF
        full_mask = (1 << k_val) - 1
        # Lastly, add the free distance from the endpoint of the final forced edge to island N.
        for i in range(k_val):
            for o in (0, 1):
                cand = dp[full_mask][i][o] + dist[ forced_opts[i][o][1] ][N-1]
                if cand < best:
                    best = cand
        out_lines.append(str(best))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()