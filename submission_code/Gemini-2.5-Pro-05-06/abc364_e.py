import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes_input = []
    for _ in range(N):
        dishes_input.append(list(map(int, sys.stdin.readline().split())))

    # dp[k][sA] stores the minimum saltiness after Snuke has eaten k dishes,
    # such that these k dishes resulted in a total sweetness of sA,
    # AND both sA <= X and the minimum saltiness dp[k][sA] <= Y.
    # This state represents a valid prefix sequence.
    dp = [[Y + 1] * (X + 1) for _ in range(N + 1)]
    dp[0][0] = 0  # Base case: 0 dishes, 0 sweetness, 0 saltiness.
    
    ans = 0 # Stores the maximum number of dishes Snuke can eat.

    for dish_idx in range(N):
        a_curr, b_curr = dishes_input[dish_idx]
        
        # Iterate k_prev downwards: number of dishes already eaten satisfying limits.
        # Iterate sA_prev downwards: total sweetness for these k_prev dishes.
        # This order ensures each dish is considered at most once (0/1 knapsack property).
        for k_prev in range(N - 1, -1, -1):
            for sA_prev in range(X, -1, -1):
                
                # Check if state (k_prev dishes, sA_prev sweetness) is reachable and valid.
                # sA_prev <= X is ensured by loop range.
                # dp[k_prev][sA_prev] <= Y ensures saltiness is within limits.
                if dp[k_prev][sA_prev] <= Y:
                    # This state is a valid prefix. Snuke has eaten k_prev dishes
                    # and current totals (sA_prev, dp[k_prev][sA_prev]) are within limits.
                    
                    # Now, Snuke attempts to eat the current dish (a_curr, b_curr).
                    # This becomes the (k_prev + 1)-th dish eaten.
                    # Update the maximum number of dishes eaten so far.
                    # This is a candidate for the answer, even if limits are now exceeded.
                    ans = max(ans, k_prev + 1)
                    
                    # Calculate new total sweetness and saltiness.
                    sA_new = sA_prev + a_curr
                    sB_new = dp[k_prev][sA_prev] + b_curr
                    
                    # If this new state (k_prev + 1 dishes, totals sA_new, sB_new)
                    # ALSO satisfies the limits (sA_new <= X and sB_new <= Y),
                    # then it can serve as a valid prefix for eating even more dishes.
                    # Update the dp table for (k_prev + 1) dishes.
                    if sA_new <= X and sB_new <= Y:
                        dp[k_prev+1][sA_new] = min(dp[k_prev+1][sA_new], sB_new)
    
    print(ans)

solve()