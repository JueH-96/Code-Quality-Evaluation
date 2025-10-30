import sys

def solve():
    # Read N (number of vertices) and M (number of edges)
    N, M = map(int, sys.stdin.readline().split())

    # Initialize adjacency list (1-indexed for convenience)
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Read weights W_i (1-indexed)
    # W[0] is unused to keep indices aligned with problem description
    W = [0] + list(map(int, sys.stdin.readline().split())) 

    # Read initial pieces A_i (1-indexed)
    # A[0] is unused
    A = [0] + list(map(int, sys.stdin.readline().split())) 

    # Create a list of (weight, vertex_index) pairs
    # Sorting this list will allow processing vertices in increasing order of their W values
    vertices_sorted_by_W = []
    for i in range(1, N + 1):
        vertices_sorted_by_W.append((W[i], i))
    vertices_sorted_by_W.sort()

    # f[v] will store the maximum number of operations a single piece at vertex v can generate
    f = [0] * (N + 1)

    # Iterate through vertices sorted by their weights
    for _, v in vertices_sorted_by_W:
        # knapsack_dp[k] will store the maximum sum of f[y] values for a set S
        # whose total weight sum(W_y for y in S) is exactly k.
        # The capacity for the knapsack is W[v] - 1.
        # So, the array needs to be of size W[v] to cover indices from 0 to W[v]-1.
        knapsack_dp = [0] * W[v] 
        
        # Collect items (item_W, item_f) for the knapsack
        # An item corresponds to a neighbor 'u' of 'v' such that W[u] < W[v].
        # item_W = W[u], item_f = f[u]
        items_for_knapsack = []
        for u in adj[v]:
            if W[u] < W[v]:
                items_for_knapsack.append((W[u], f[u]))
        
        # Perform 0/1 knapsack calculation for vertex v
        # Iterate through each available item
        for item_W, item_f in items_for_knapsack:
            # Iterate through knapsack capacities in reverse order.
            # This ensures that each item is considered at most once (0/1 knapsack property).
            # The range is from (W[v] - 1) down to item_W.
            for current_capacity in range(W[v] - 1, item_W - 1, -1):
                knapsack_dp[current_capacity] = max(
                    knapsack_dp[current_capacity],
                    knapsack_dp[current_capacity - item_W] + item_f
                )
        
        # After filling the knapsack_dp array, find the maximum value among all possible capacities.
        # This max value is the maximum sum of f[y] from neighbors.
        max_sum_f_y = 0
        if W[v] > 1: # If W[v] is 1, capacity is 0, knapsack_dp is [0], max_sum_f_y should be 0.
            max_sum_f_y = max(knapsack_dp)
        
        # The total operations from a piece at 'v' is 1 (for removing it) + the max_sum_f_y from new pieces.
        f[v] = 1 + max_sum_f_y

    # Calculate the total maximum operations by summing A_i * f[i] for all vertices.
    total_operations = 0
    for i in range(1, N + 1):
        total_operations += A[i] * f[i]

    # Print the final result
    print(total_operations)

# Call the solve function to execute the program
solve()