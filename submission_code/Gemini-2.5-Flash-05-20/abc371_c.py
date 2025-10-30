import itertools

def solve():
    N = int(input())

    # Initialize adjacency matrix for graph G (1-based indexing)
    G_adj = [[False] * (N + 1) for _ in range(N + 1)]
    M_G = int(input())
    for _ in range(M_G):
        u, v = map(int, input().split())
        G_adj[u][v] = G_adj[v][u] = True

    # Initialize adjacency matrix for graph H (1-based indexing)
    H_adj = [[False] * (N + 1) for _ in range(N + 1)]
    M_H = int(input())
    for _ in range(M_H):
        a, b = map(int, input().split())
        H_adj[a][b] = H_adj[b][a] = True

    # Initialize cost matrix A (1-based indexing, symmetric)
    A_cost = [[0] * (N + 1) for _ in range(N + 1)]
    all_costs_flat = []
    # Read costs A_ij for 1 <= i < j <= N
    # The input format provides costs row by row:
    # A_{1,2} A_{1,3} ... A_{1,N}
    # A_{2,3} ... A_{2,N}
    # ...
    # A_{N-1,N}
    for i in range(1, N): # Iterate N-1 times for rows corresponding to i=1 to N-1
        line_costs = list(map(int, input().split()))
        all_costs_flat.extend(line_costs)
    
    # Populate the A_cost matrix from the flattened list
    cost_idx = 0
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            A_cost[i][j] = A_cost[j][i] = all_costs_flat[cost_idx]
            cost_idx += 1

    min_total_cost = float('inf') # Initialize with a very large value

    # Iterate through all permutations of vertices (1, ..., N)
    # P_tuple represents a permutation (P_1, P_2, ..., P_N), 
    # where G's vertex k maps to H's vertex P_k.
    for P_tuple in itertools.permutations(range(1, N + 1)):
        current_total_cost = 0
        
        # Iterate over all unique pairs of vertices (i, j) in G (1-based indexing)
        # We only need to check pairs where i < j to avoid duplicates and self-loops
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                # Check if an edge exists between G's vertex i and j
                g_has_edge = G_adj[i][j]

                # Map G's vertices (i, j) to H's vertices using the current permutation P
                # P_tuple is 0-indexed, so P_k corresponds to P_tuple[k-1]
                h_vertex_1 = P_tuple[i - 1]
                h_vertex_2 = P_tuple[j - 1]

                # Check if an edge currently exists between h_vertex_1 and h_vertex_2 in H
                h_has_edge_permuted = H_adj[h_vertex_1][h_vertex_2]

                # If the existence of the edge in G differs from H's corresponding edge,
                # an operation (add or remove) is required.
                if g_has_edge != h_has_edge_permuted:
                    # Add the cost of this operation.
                    # The cost A_{u,v} is the same whether adding or removing.
                    current_total_cost += A_cost[h_vertex_1][h_vertex_2]
        
        # Update the minimum total cost found so far
        min_total_cost = min(min_total_cost, current_total_cost)

    print(min_total_cost)

# Call the solve function to run the program
solve()