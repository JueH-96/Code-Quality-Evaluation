import math

def dist_manual(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx*dx + dy*dy)

def solve():
    N = int(input())
    coords = []
    for _ in range(N):
        x, y = map(int, input().split())
        coords.append((x, y))

    # DP[k_v_parity][i]: minimum distance to reach checkpoint i (0-indexed)
    # using a path with a number of vertices whose parity is k_v_parity.
    # We iterate over the number of vertices k_v from 1 to N.
    # We only need DP values from the previous number of vertices (k_v-1) to calculate
    # values for the current number of vertices (k_v). This allows O(N) memory.
    # DP table size: 2 x N. DP[parity][i].

    # Initialize DP table with infinity. DP[parity][i].
    # k_v from 1 to N. Parity 1: 1, 3, 5... Parity 0: 2, 4, 6...
    DP = [[float('inf')] * N for _ in range(2)]

    # Base case: reaching checkpoint 0 with 1 vertex (the path is just 0)
    # k_v = 1, parity = 1 % 2 = 1.
    DP[1][0] = 0.0

    # Iterate over number of vertices k_v in the path
    for k_v in range(2, N + 1):
        curr_parity = k_v % 2
        prev_parity = (k_v - 1) % 2

        # Initialize the current layer DP[curr_parity] with infinity
        for i in range(N):
            DP[curr_parity][i] = float('inf')

        # Iterate over current checkpoint i (the k_v-th vertex in the path)
        # Since path is 0=p_0 < p_1 < ... < p_{k_v-1}=i, i must be >= k_v-1.
        for i in range(k_v - 1, N):
            # Iterate over previous checkpoint p (the (k_v-1)-th vertex in the path)
            # p must be < i.
            # p must be >= k_v-2 (since p is the (k_v-1)-th vertex, p >= (k_v-1) - 1 = k_v-2).
            for p in range(k_v - 2, i):
                if DP[prev_parity][p] != float('inf'):
                    dist_p_i = dist_manual(coords[p], coords[i])
                    DP[curr_parity][i] = min(DP[curr_parity][i], DP[prev_parity][p] + dist_p_i)

    # Calculate the minimum total cost
    min_total_cost = float('inf')

    # The path must end at checkpoint N-1.
    # Consider paths reaching N-1 with k_v vertices, where k_v ranges from 2 to N.
    for k_v in range(2, N + 1):
        curr_parity = k_v % 2
        
        if DP[curr_parity][N - 1] != float('inf'):
            distance_traveled = DP[curr_parity][N - 1]

            # Number of intermediate checkpoints visited (between checkpoint 1 and N)
            # Using 0-indexed, between checkpoint 0 and N-1.
            # Intermediate checkpoints are 1, 2, ..., N-2. Total N-2.
            # Path has k_v vertices: 0=p_0, p_1, ..., p_{k_v-1}=N-1.
            # The intermediate vertices in the path are p_1, ..., p_{k_v-2}. There are k_v - 2 of them.
            num_intermediate_visited = k_v - 2

            # Total number of intermediate checkpoints available: N-2.
            num_intermediate_total = N - 2

            # Number of skipped intermediate checkpoints C
            num_skipped = num_intermediate_total - num_intermediate_visited

            current_penalty = 0.0
            if num_skipped > 0:
                current_penalty = 2.0**(num_skipped - 1)

            current_cost = distance_traveled + current_penalty
            min_total_cost = min(min_total_cost, current_cost)

    print(f"{min_total_cost:.17f}") # Using 17 decimal places for output

solve()