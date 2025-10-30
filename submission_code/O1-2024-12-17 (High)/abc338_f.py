def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster local "index" usage rather than repeated pop(0):
    # We'll parse integers from input_data with an index pointer.
    idx = 0
    def read_int():
        nonlocal idx
        val = int(input_data[idx])
        idx += 1
        return val

    N = read_int()
    M = read_int()
    
    INF = 10**18
    
    # Initialize distance matrix for Floyd-Warshall
    dist = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    
    for _ in range(M):
        u = read_int() - 1
        v = read_int() - 1
        w = read_int()
        dist[u][v] = w
    
    # Floyd-Warshall to get all-pairs shortest paths, accounting for possible negative edges
    # (the problem guarantees no negative cycles).
    for k in range(N):
        for i in range(N):
            if dist[i][k] == INF:
                continue
            ik = dist[i][k]
            row_k = dist[k]
            row_i = dist[i]
            for j in range(N):
                # If dist[k][j] is also finite, try updating dist[i][j]
                tmp = ik + row_k[j]
                if tmp < row_i[j]:
                    row_i[j] = tmp
    
    # We now do a TSP-like DP on the complete "shortest-path" graph:
    # dp[mask][v] = minimum cost of a route that visits exactly
    # the vertices in "mask" once and ends at vertex v.
    # Because "visiting all vertices at least once" in the original graph
    # corresponds to a Hamiltonian path in the shortest-path graph.
    
    # We'll store dp in a single list to reduce indexing overhead:
    # dp_index(mask, v) = mask*N + v
    dp_size = (1 << N) * N
    dp = [INF]*dp_size
    
    # Initialize: starting a path at each vertex v with only v visited has cost = 0
    for v in range(N):
        dp[(1 << v) * N + v] = 0
    
    # Precompute, for each v, the list of w for which dist[v][w] < INF
    # (i.e., w is reachable from v)
    adj = []
    for v in range(N):
        rowv = dist[v]
        adj.append([w for w in range(N) if rowv[w] < INF])
    
    full_mask = (1 << N) - 1
    
    # Held-Karp DP: O(N^2 * 2^N) in the worst case
    for mask in range(1 << N):
        base = mask * N
        for v in range(N):
            cost_v = dp[base + v]
            if cost_v == INF:
                continue
            # Try extending the path to visit a new vertex w not in mask
            rowv = dist[v]
            for w in adj[v]:
                if (mask & (1 << w)) == 0:
                    new_mask = mask | (1 << w)
                    idx_w = new_mask * N + w
                    val = cost_v + rowv[w]
                    if val < dp[idx_w]:
                        dp[idx_w] = val
    
    # The answer is the minimum dp over all endpoints v, with mask = full_mask
    ans = INF
    end_base = full_mask * N
    for v in range(N):
        val = dp[end_base + v]
        if val < ans:
            ans = val
    
    if ans == INF:
        print("No")
    else:
        print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()