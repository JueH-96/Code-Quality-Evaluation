import sys

# Set a sufficiently large infinity value for path costs.
# float('inf') is suitable as Python handles large integers automatically.
INF = float('inf')

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Store bridge information. bridges_info[i] corresponds to bridge (i+1).
    bridges_info = []
    for _ in range(M):
        u, v, t = map(int, sys.stdin.readline().split())
        bridges_info.append((u, v, t))

    # Precompute all-pairs shortest paths using Floyd-Warshall.
    # dist[i][j] will store the shortest time to travel between island i and island j
    # using any available bridges (not specifically the 'required' ones).
    dist = [[INF] * (N + 1) for _ in range(N + 1)]

    # Initialize distances: 0 for self-loops, direct bridge costs for connected islands.
    for i in range(1, N + 1):
        dist[i][i] = 0

    for u, v, t in bridges_info:
        # Take the minimum cost if multiple bridges connect the same pair of islands
        dist[u][v] = min(dist[u][v], t)
        dist[v][u] = min(dist[v][u], t)

    # Floyd-Warshall algorithm
    for k in range(1, N + 1):  # Intermediate node
        for i in range(1, N + 1):  # Start node
            for j in range(1, N + 1):  # End node
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    Q = int(sys.stdin.readline())

    results = []
    for _ in range(Q):
        K = int(sys.stdin.readline())
        # Convert 1-indexed bridge IDs to 0-indexed for list access
        B_indices = [idx - 1 for idx in map(int, sys.stdin.readline().split())]

        required_bridges = [bridges_info[idx] for idx in B_indices]

        # Identify all 'special' nodes for this query: island 1, island N, and endpoints of required bridges.
        special_nodes_set = {1, N}
        for u, v, _ in required_bridges:
            special_nodes_set.add(u)
            special_nodes_set.add(v)
        
        # Convert set of special nodes to a sorted list and create a mapping for compact DP table indices.
        P = sorted(list(special_nodes_set))
        node_to_idx = {node: i for i, node in enumerate(P)}
        num_special_nodes = len(P)

        # dp[mask][p_idx] = minimum cost to have visited bridges in 'mask'
        # and currently be at the island P[p_idx].
        dp = [[INF] * num_special_nodes for _ in range(1 << K)]

        # Base case: Starting at island 1, no bridges used, cost 0.
        dp[0][node_to_idx[1]] = 0

        # Iterate through all possible masks (subsets of required bridges used)
        for mask in range(1 << K):
            # Iterate through all possible current special nodes
            for p_idx1 in range(num_special_nodes):
                u = P[p_idx1]
                
                # If current state is unreachable, skip
                if dp[mask][p_idx1] == INF:
                    continue

                # Try to extend the path by using an unvisited required bridge 'j'
                for j in range(K):
                    if not ((mask >> j) & 1):  # If bridge j is not yet used
                        U_j, V_j, T_j = required_bridges[j]

                        # Option 1: Travel from current island 'u' to U_j (shortest path),
                        # then traverse bridge j from U_j to V_j.
                        # Since the graph is connected, dist[u][U_j] will always be finite.
                        new_cost = dp[mask][p_idx1] + dist[u][U_j] + T_j
                        new_mask = mask | (1 << j)
                        target_node_idx = node_to_idx[V_j]
                        dp[new_mask][target_node_idx] = min(dp[new_mask][target_node_idx], new_cost)

                        # Option 2: Travel from current island 'u' to V_j (shortest path),
                        # then traverse bridge j from V_j to U_j.
                        # Since the graph is connected, dist[u][V_j] will always be finite.
                        new_cost = dp[mask][p_idx1] + dist[u][V_j] + T_j
                        new_mask = mask | (1 << j)
                        target_node_idx = node_to_idx[U_j]
                        dp[new_mask][target_node_idx] = min(dp[new_mask][target_node_idx], new_cost)
        
        # After filling the DP table, find the minimum total cost.
        # This is the minimum cost to have used all K bridges (final_mask)
        # and then travel from the last special node to island N.
        min_total_cost = INF
        final_mask = (1 << K) - 1 # Mask where all K bits are set

        for p_idx_final in range(num_special_nodes):
            u_final = P[p_idx_final]
            if dp[final_mask][p_idx_final] != INF:
                # Add the cost to travel from the last visited special node (u_final) to island N.
                # Since the graph is connected, dist[u_final][N] will always be finite.
                min_total_cost = min(min_total_cost, dp[final_mask][p_idx_final] + dist[u_final][N])
        
        results.append(min_total_cost)
    
    # Print all query results
    for res in results:
        sys.stdout.write(str(res) + '
')

# Call the solve function to execute the program
solve()