def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    n, m = map(int, input_data[:2])
    edges = input_data[2:]
    
    INF = 10**18
    
    # Step 1: Initialize the adjacency matrix for Floyd-Warshall
    dist = [[INF]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    
    idx = 0
    for _ in range(m):
        u = int(edges[idx]); v = int(edges[idx+1]); w = int(edges[idx+2])
        idx += 3
        dist[u-1][v-1] = min(dist[u-1][v-1], w)  # in case multiple edges or just set
    
    # Step 2: Floyd-Warshall to get all-pairs shortest paths
    # This accounts for any negative edges but no negative cycles (as guaranteed).
    for k in range(n):
        for i in range(n):
            # Small optimization: skip if dist[i][k] is INF
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[k][j] == INF:
                    continue
                # Relaxation
                nd = dist[i][k] + dist[k][j]
                if nd < dist[i][j]:
                    dist[i][j] = nd
    
    # Check if there's any vertex that cannot be reached from some other vertex.
    # Actually, we only need that we can pairwise travel among all vertices in some sequence.
    # If for some pair (i, j), dist[i][j] == INF, it does not necessarily doom the entire solution,
    # but in a TSP sense, we must be able to pick a route that visits all vertices. If there exists
    # a vertex j that is unreachable from every i, there's no valid walk. We'll detect that
    # in the TSP DP because transitions will be impossible.
    
    # Step 3: TSP DP: dp[mask][i] = minimal cost to have visited vertices in mask (bitmask),
    #                ending at vertex i.
    # We'll compute dp for mask in ascending order of bits. Complexity: O(n^2 2^n) which is
    # borderline for n=20 in Python, but we'll try to implement efficiently.
    
    # If it turns out to be too slow in practice, an optimized approach or pruning might be needed.
    # We'll do our best with standard bitmask DP.
    
    # Initialize DP
    # dp[mask][i] = INF  for all mask, i
    dp = [[INF]*n for _ in range(1<<n)]
    # Base cases
    for i in range(n):
        dp[1 << i][i] = 0  # Starting "path" with only i visited, ended at i, cost=0
    
    # Precompute to speed up iteration: for a given mask, the unvisited vertices
    # so we don't gather them each time with heavy loops.
    
    for mask in range(1<<n):
        # We will iterate over the end-vertex i that is in mask
        # Then we try to go to j not in mask
        # dp[mask | (1<<j)][j] = min of dp[mask][i] + dist[i][j]
        # The standard TSP transition.
        # We first collect the vertices in "mask" (endpoints i) to iterate over.
        
        # Trick: to iterate i in mask, we do a sub-loop:
        #    i = subset
        #    while i > 0:
        #      i_trailing = i & -i  # get lowest set bit
        #      i_vertex = index_of(i_trailing)
        #      ...
        #      i &= (i-1)
        #
        # but indexing "i_vertex" is the tricky part. Or we can do a linear check from 0..n-1.
        # Since n=20, we can just do a linear check.
        
        # Let's skip if mask has no endpoints in practice, but it always does except mask=0.
        # We'll do it anyway more simply:
        
        for i in range(n):
            if not ((mask >> i) & 1):
                continue
            cost_i = dp[mask][i]
            if cost_i == INF:
                continue
            # Now let's try to go to j
            nxt = (~mask) & ((1 << n) - 1)  # the set of vertices not in mask
            # We'll iterate j in [0..n-1], check if j not in mask.
            # Then update dp.
            
            for j in range(n):
                if (nxt >> j) & 1:  # means j not in mask
                    c = dist[i][j]
                    if c == INF:
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = cost_i + c
                    if new_cost < dp[new_mask][j]:
                        dp[new_mask][j] = new_cost
    
    full_mask = (1 << n) - 1
    ans = INF
    for i in range(n):
        val = dp[full_mask][i]
        if val < ans:
            ans = val
    
    if ans >= INF:
        print("No")
    else:
        print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()