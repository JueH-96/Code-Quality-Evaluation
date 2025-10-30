import heapq

def solve():
    N, M = map(int, input().split())
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
    
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))

    # The problem asks for the maximum number of operations.
    # Let pi_i be the maximum number of additional operations one piece at vertex i can enable.
    # When a piece is removed from i, it counts as 1 operation.
    # If a set S of neighbors is chosen (sum W_y < W_i), one piece is placed on each y in S.
    # Each piece on y can enable pi_y additional operations.
    # So, pi_i = 1 (for removing the piece at i) + max_{S subset N(i), sum W_y < W_i} (sum_{y in S} pi_y).
    # The base case is when only S=empty is possible (sum W_y < W_i is only true for S=empty), in which case sum_{y in S} pi_y = 0.
    # This happens if W_i = 1 (as W_y >= 1, sum W_y >= 1 for non-empty S) or if min_{y in N(i)} W_y >= W_i.

    # We can compute pi values using dynamic programming or fixed-point iteration.
    # The dependencies are pi_i depends on pi_j for neighbors j. This can involve cycles.
    # However, the constraint sum W_y < W_i is key. When maximizing sum pi_y, it's a knapsack-like problem.
    # To maximize sum pi_y under sum W_y < W_i, we should pick neighbors with high pi_y values
    # as long as their total weight W_y does not exceed W_i-1.
    # This suggests a knapsack DP where value is pi_y and weight is W_y, capacity is W_i-1.
    # The DP state for vertex i needs pi values of its neighbors.

    # We can use fixed-point iteration. Start with initial pi values (e.g., pi_i = 1, representing just the removal operation).
    # In each iteration, update pi_i using the recursive formula based on pi values from the previous iteration.
    # Since the total weight decreases in each operation that produces pieces, the pi values should converge.
    # The maximum number of iterations for values to propagate in a graph of N vertices could be N.

    # Initialize pi values. A piece at least enables 1 operation (its own removal).
    pi = [1] * N

    # Iterate N+1 times to ensure convergence.
    for _ in range(N + 1):
        new_pi = [0] * N
        for i in range(N):
            capacity = W[i] - 1
            
            if capacity < 0: # W[i] == 0, should not happen based on constraints W_i >= 1.
                 capacity = 0 # If W[i] = 1, capacity is 0. Only S={} is possible.
            
            neighbors_info = []
            for j in adj[i]:
                neighbors_info.append((W[j], pi[j])) # Neighbor weight and current pi value

            # Solve 0/1 knapsack-like problem: select subset of neighbors to maximize sum pi_j
            # under total weight sum <= capacity.
            # Items: neighbors. Item weight: W[j]. Item value: pi[j]. Knapsack capacity: capacity.

            # dp[w] = maximum sum of pi_y for a subset S with sum W_y = w.
            dp = [0] * (capacity + 1)

            for neighbor_weight, neighbor_pi in neighbors_info:
                # Iterate DP table backwards
                for w in range(capacity, neighbor_weight - 1, -1):
                    dp[w] = max(dp[w], dp[w - neighbor_weight] + neighbor_pi)

            max_neighbor_pi_sum = dp[capacity]
            new_pi[i] = 1 + max_neighbor_pi_sum

        pi = new_pi # Update pi values for the next iteration

    # The total number of operations is the sum of initial pieces multiplied by the maximum
    # number of operations each piece can enable (its pi value).
    
    total_operations = 0
    for i in range(N):
        total_operations += A[i] * pi[i]

    print(total_operations)

solve()