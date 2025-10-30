import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = []
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        dishes.append((A, B))

    # MAX_SWEETNESS_SUM defines the maximum possible sweetness index in our DP table.
    # The sum can go up to X (current limit) plus the sweetness of the last dish (max 10000).
    # So, X + 10000 is a safe upper bound for sweetness sum that can be "counted".
    MAX_SWEETNESS_SUM = X + 10000 

    # dp[k][s] stores the minimum total saltiness for exactly k dishes
    # that result in s total sweetness.
    # Initialize with float('inf') to represent unreachable states.
    dp = [[float('inf')] * (MAX_SWEETNESS_SUM + 1) for _ in range(N + 1)]
    dp[0][0] = 0 # Base case: 0 dishes, 0 sweetness, 0 saltiness

    # Iterate through each dish
    for a, b in dishes:
        # Iterate k_prev from N-1 down to 0, representing the number of dishes already selected.
        # Iterate s_prev from MAX_SWEETNESS_SUM down to 0, representing the sweetness sum of k_prev dishes.
        # Iterating downwards prevents using the current dish multiple times in the same outer loop iteration.
        for k_prev in range(N - 1, -1, -1):
            for s_prev in range(MAX_SWEETNESS_SUM, -1, -1):
                # If the state (k_prev, s_prev) is unreachable, skip it.
                if dp[k_prev][s_prev] == float('inf'):
                    continue

                # IMPORTANT: Apply the "at any point" condition.
                # If the current cumulative sums (s_prev, dp[k_prev][s_prev]) already exceed X or Y,
                # Snuke would have stopped *before* eating these k_prev dishes or he stops
                # after the dish that caused exceedance but before eating *further* dishes.
                # The interpretation that matches sample 1 output is:
                # To extend a path (add another dish), the *previous* state must have been valid (<= X and <= Y).
                if s_prev > X or dp[k_prev][s_prev] > Y:
                    continue

                # Calculate new sums and dish count after adding the current dish (a, b)
                s_curr = s_prev + a
                b_curr = dp[k_prev][s_prev] + b
                k_curr = k_prev + 1

                # Ensure s_curr is within the bounds of our DP table index.
                # If it exceeds, it means this path cannot be stored.
                if s_curr > MAX_SWEETNESS_SUM:
                    continue
                
                # Update the DP table: Take the minimum saltiness for this new state (k_curr, s_curr)
                dp[k_curr][s_curr] = min(dp[k_curr][s_curr], b_curr)

    max_dishes = 0
    # After all dishes have been processed, find the maximum number of dishes (k)
    # for which a valid state (k, s) is reachable.
    # A state (k, s) is reachable if dp[k][s] is not float('inf').
    # The total sweetness 's' and saltiness 'dp[k][s]' for this state can exceed X or Y
    # because the k-th dish is counted even if it pushes the sums over the limit.
    for k in range(N + 1):
        for s in range(MAX_SWEETNESS_SUM + 1):
            if dp[k][s] != float('inf'):
                max_dishes = max(max_dishes, k)
                
    sys.stdout.write(str(max_dishes) + "
")

solve()