import sys

def solve():
    """
    Solves the "Travel on Islands" problem by combining Floyd-Warshall for all-pairs
    shortest paths with a TSP-style dynamic programming approach for each query.
    """
    
    # Use a large number for infinity
    INF = float('inf')

    # --- 1. Read Graph and Pre-compute All-Pairs Shortest Paths ---
    
    # Fast I/O
    input = sys.stdin.readline
    
    # Read graph dimensions
    try:
        N, M = map(int, input().split())
    except (IOError, ValueError):
        return # Handle empty input at the start
        
    # Store all bridges (1-indexed for convenience, bridges[0] is unused)
    bridges = [None]
    for _ in range(M):
        u, v, t = map(int, input().split())
        bridges.append((u, v, t))

    # Initialize distance matrix for Floyd-Warshall (1-based indexing for islands)
    dist = [[INF] * (N + 1) for _ in range(N + 1)]

    for i in range(1, N + 1):
        dist[i][i] = 0

    for u, v, t in bridges[1:]:
        dist[u][v] = min(dist[u][v], t)
        dist[v][u] = min(dist[v][u], t)

    # Run Floyd-Warshall algorithm
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # --- 2. Process Queries ---
    
    try:
        Q = int(input())
    except (IOError, ValueError):
        return # Handle case where there are no queries
        
    for _ in range(Q):
        line = list(map(int, input().split()))
        K = line[0]
        required_bridge_indices = line[1:]

        # Get details of required bridges for the current query
        required_bridges = [bridges[b_idx] for b_idx in required_bridge_indices]

        # --- 3. DP for TSP on Required Bridges ---
        # dp[mask][last_idx][endpoint_choice]
        # state: (mask of visited bridges, index of last bridge, which endpoint of last bridge)
        # endpoint_choice: 0 for the first listed endpoint (U), 1 for the second (V)
        dp = [[[INF] * 2 for _ in range(K)] for _ in range(1 << K)]

        START_NODE = 1
        END_NODE = N

        # Initialization (DP base cases: paths visiting one required bridge)
        for i in range(K):
            Ui, Vi, Ti = required_bridges[i]
            mask = 1 << i
            
            # Path: START_NODE -> Ui, cross to Vi, ending at Vi
            if dist[START_NODE][Ui] != INF:
                dp[mask][i][1] = dist[START_NODE][Ui] + Ti
            # Path: START_NODE -> Vi, cross to Ui, ending at Ui
            if dist[START_NODE][Vi] != INF:
                dp[mask][i][0] = dist[START_NODE][Vi] + Ti

        # DP transitions
        for mask in range(1, 1 << K):
            for j in range(K):
                if not ((mask >> j) & 1):
                    continue
                # We are calculating costs for paths of set 'mask' ending at bridge 'j'
                Uj, Vj, Tj = required_bridges[j]
                
                prev_mask = mask ^ (1 << j)
                if prev_mask == 0:
                    continue

                for i in range(K):
                    if not ((prev_mask >> i) & 1):
                        continue
                    # Consider transitioning from a path ending at bridge 'i'
                    Ui, Vi, _ = required_bridges[i]

                    # Case 1: Previous path ended at Ui
                    if dp[prev_mask][i][0] != INF:
                        if dist[Ui][Uj] != INF: # Ui -> Uj -> Vj
                            dp[mask][j][1] = min(dp[mask][j][1], dp[prev_mask][i][0] + dist[Ui][Uj] + Tj)
                        if dist[Ui][Vj] != INF: # Ui -> Vj -> Uj
                            dp[mask][j][0] = min(dp[mask][j][0], dp[prev_mask][i][0] + dist[Ui][Vj] + Tj)
                    
                    # Case 2: Previous path ended at Vi
                    if dp[prev_mask][i][1] != INF:
                        if dist[Vi][Uj] != INF: # Vi -> Uj -> Vj
                            dp[mask][j][1] = min(dp[mask][j][1], dp[prev_mask][i][1] + dist[Vi][Uj] + Tj)
                        if dist[Vi][Vj] != INF: # Vi -> Vj -> Uj
                            dp[mask][j][0] = min(dp[mask][j][0], dp[prev_mask][i][1] + dist[Vi][Vj] + Tj)

        # --- 4. Calculate Final Answer for the Query ---
        min_total_time = INF
        final_mask = (1 << K) - 1

        for i in range(K):
            Ui, Vi, _ = required_bridges[i]
            
            # Path ended at Ui, now travel to END_NODE
            if dp[final_mask][i][0] != INF and dist[Ui][END_NODE] != INF:
                min_total_time = min(min_total_time, dp[final_mask][i][0] + dist[Ui][END_NODE])
            
            # Path ended at Vi, now travel to END_NODE
            if dp[final_mask][i][1] != INF and dist[Vi][END_NODE] != INF:
                min_total_time = min(min_total_time, dp[final_mask][i][1] + dist[Vi][END_NODE])
        
        print(min_total_time)

solve()