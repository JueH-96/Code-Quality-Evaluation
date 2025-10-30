import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    INF = float('inf')

    # Initialize adjacency matrix for Floyd-Warshall
    # dist[i][j] stores the shortest path cost from vertex i to vertex j.
    # Vertices are 0-indexed internally.
    dist = [[INF] * N for _ in range(N)]
    
    for i in range(N):
        dist[i][i] = 0 # Cost to stay at the same vertex is 0.

    # Read M edges and populate initial dist matrix.
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        u -= 1  # Convert to 0-indexed
        v -= 1  # Convert to 0-indexed
        # Problem statement: (U_i,V_i) are distinct pairs, so one direct edge (u,v).
        # Using min is robust; dist[u][v] = w would also be fine.
        dist[u][v] = min(dist[u][v], w) 

    # Floyd-Warshall algorithm
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # DP state: dp[mask][v_last] = minimum cost to visit vertices in 'mask', ending at 'v_last'.
    dp = [[INF] * N for _ in range(1 << N)] # Dimensions: (2^N) x N

    # Base cases: A walk visiting only vertex 'i', starting and ending at 'i', has cost 0.
    for i in range(N):
        dp[1 << i][i] = 0

    # Iterate over masks (sets of visited vertices)
    for mask in range(1, 1 << N):
        # 'v_last' is the vertex where the walk covering 'mask' ends.
        for v_last in range(N):
            if not ((mask >> v_last) & 1): # If v_last is not in 'mask', skip.
                continue

            # 'prev_mask' = mask \ {v_last}
            prev_mask = mask ^ (1 << v_last)
            
            if prev_mask == 0: # If 'mask' had only 'v_last' (i.e. |mask|=1). Base case, already handled.
                continue
            
            # 'u_prev' is the vertex where the walk covering 'prev_mask' ended.
            for u_prev in range(N):
                if not ((prev_mask >> u_prev) & 1): # If u_prev is not in 'prev_mask', skip.
                    continue
                
                # If state (prev_mask, u_prev) is reachable AND path from u_prev to v_last exists:
                if dp[prev_mask][u_prev] != INF and dist[u_prev][v_last] != INF:
                    new_cost = dp[prev_mask][u_prev] + dist[u_prev][v_last]
                    dp[mask][v_last] = min(dp[mask][v_last], new_cost)
    
    # Final mask representing all N vertices visited.
    all_vertices_mask = (1 << N) - 1
    min_total_weight = INF

    # Find minimum cost among paths ending at any vertex, covering all vertices.
    for v_last in range(N):
        min_total_weight = min(min_total_weight, dp[all_vertices_mask][v_last])

    if min_total_weight == INF:
        print("No")
    else:
        # Weights are integers, so the sum will be an integer.
        print(int(min_total_weight))

solve()