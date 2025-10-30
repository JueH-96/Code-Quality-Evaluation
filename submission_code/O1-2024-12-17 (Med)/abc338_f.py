def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = input_data[2:]
    
    # Large "infinity" value for initialization
    INF = 10**18
    
    # Initialize shortest-path matrix
    # Use 0-based indexing internally
    sp = [[INF]*N for _ in range(N)]
    for i in range(N):
        sp[i][i] = 0
    
    idx = 0
    for _ in range(M):
        u = int(edges[idx]) - 1
        v = int(edges[idx+1]) - 1
        w = int(edges[idx+2])
        idx += 3
        sp[u][v] = min(sp[u][v], w)  # In case of duplicates, but problem states none
    
    # Floyd-Warshall for all-pairs shortest paths
    for k in range(N):
        for i in range(N):
            if sp[i][k] == INF: 
                continue
            tmp_ik = sp[i][k]
            for j in range(N):
                if sp[k][j] == INF:
                    continue
                cand = tmp_ik + sp[k][j]
                if cand < sp[i][j]:
                    sp[i][j] = cand
    
    # Now use Held-Karp DP for TSP (but no need to return to start).
    # dp[mask][v]: minimum cost to visit all vertices in 'mask' and end at vertex v
    dp = [[INF]*N for _ in range(1<<N)]
    
    # Base cases: start at each vertex
    for v in range(N):
        dp[1 << v][v] = 0
    
    # Build up masks
    for mask in range(1<<N):
        # For each possible "last" vertex
        for v in range(N):
            cost_v = dp[mask][v]
            if cost_v == INF:
                continue
            # Try to go to a next vertex w not in mask
            for w in range(N):
                if (mask & (1 << w)) == 0:
                    # There's a direct shortest-path v->w if sp[v][w] != INF
                    if sp[v][w] == INF:
                        continue
                    new_mask = mask | (1 << w)
                    new_cost = cost_v + sp[v][w]
                    if new_cost < dp[new_mask][w]:
                        dp[new_mask][w] = new_cost
    
    # The answer is the minimum cost among dp[(1<<N)-1][v] for v in [0..N-1]
    ans = min(dp[(1<<N)-1][v] for v in range(N))
    
    if ans >= INF:
        print("No")
    else:
        print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()