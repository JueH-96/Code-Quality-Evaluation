import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u - 1].append(v - 1)
        adj[v - 1].append(u - 1)

    # W_list stores weights of vertices W_0, ..., W_{N-1}
    W_list = list(map(int, sys.stdin.readline().split()))
    # A stores initial piece counts A_0, ..., A_{N-1}
    A = list(map(int, sys.stdin.readline().split()))

    # dp[i] will store the maximum number of operations 
    # that can be performed starting from a single piece on vertex i.
    # Initialize with 1 (base operation: remove piece, choose S=empty)
    dp = [1] * N 

    # Sort vertices by their weights W_i.
    # Store (weight, original_index) to process vertices in increasing order of weights.
    indexed_vertices = []
    for i in range(N):
        indexed_vertices.append((W_list[i], i))
    
    # Sorts by weight, then by original_index if weights are equal (default tuple sorting)
    indexed_vertices.sort()
    
    for i in range(N):
        # Current vertex is x_idx, its weight is weight_x
        weight_x, x_idx = indexed_vertices[i]
        
        # Knapsack capacity: sum of weights of chosen neighbors must be < weight_x
        # So, sum of weights <= weight_x - 1
        capacity = weight_x - 1
        
        # If weight_x is 1, capacity is 0. 
        # The kdp_table will be [0]. Max value from knapsack is kdp_table[0] = 0.
        # Then dp[x_idx] = 1 + 0 = 1. This is correct, as no neighbors can be chosen.
        # This case is handled correctly by the general logic if capacity >= 0.
        # Since W_i >= 1, capacity is always >= 0.

        # kdp_table[k] = max_value using items with total_weight <= k
        # Size of kdp_table is capacity + 1
        kdp_table = [0] * (capacity + 1)
        
        # Iterate over neighbors of x_idx to find candidate items for knapsack
        for neighbor_original_idx in adj[x_idx]:
            # Only consider neighbors y such that W[y] < W[x]
            if W_list[neighbor_original_idx] < weight_x:
                # This neighbor is an item for the knapsack
                item_weight = W_list[neighbor_original_idx]
                # Its value is dp[neighbor_original_idx], which is already computed
                # because W_list[neighbor_original_idx] < weight_x
                item_value = dp[neighbor_original_idx]
                
                # Standard knapsack dynamic programming update:
                # Iterate capacities downwards to ensure each item is used at most once
                for j in range(capacity, item_weight - 1, -1):
                    kdp_table[j] = max(kdp_table[j], kdp_table[j - item_weight] + item_value)
        
        # dp[x_idx] = 1 (for removing the piece from x_idx itself) 
        #           + maximum value obtained from placing pieces on neighbors (kdp_table[capacity])
        dp[x_idx] = 1 + kdp_table[capacity]

    # Calculate total operations based on initial piece counts
    total_operations = 0
    for i in range(N):
        total_operations += A[i] * dp[i]
        
    print(total_operations)

solve()