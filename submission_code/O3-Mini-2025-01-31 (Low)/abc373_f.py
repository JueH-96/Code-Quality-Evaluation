def main():
    import sys, math
    input = sys.stdin.readline
    N, W = map(int, input().split())
    
    # dp[j] = maximum happiness achievable with total weight exactly j
    dp = [-10**18] * (W+1)
    dp[0] = 0
    
    # Process each item type (group)
    for _ in range(N):
        w, v = map(int, input().split())
        # For each type, if we take k items, then
        #   total weight: k * w
        #   total happiness: k * v - k^2
        #
        # Notice that the function f(k) = k*v - k^2 is concave in k.
        # Its incremental value going from k -> k+1 is
        #    f(k+1) - f(k) = v - (2k+1).
        # So, if v - (2k+1) is non-positive then taking one more item is not beneficial.
        # Solve: v - (2k+1) > 0  => k < (v-1)/2.
        # Hence, taking items beyond k = ceil(v/2) brings non-positive increments.
        #
        # Also, we cannot exceed the available weight capacity so that k <= floor(W/w).
        # Hence, we set the choice limit as:
        k_max_by_weight = W // w
        beneficial_limit = (v + 1) // 2  # equals ceil(v/2)
        limit = min(k_max_by_weight, beneficial_limit)
        
        # Build the options for this type as a list of (cost, happiness) pairs.
        choices = []
        # k == 0 is always an option.
        for k in range(limit + 1):
            cost = k * w
            happiness = k * v - k * k
            choices.append((cost, happiness))
        
        # Now, perform the group-knapsack update with these choices.
        # We use a copy (new_dp) to combine the options from previous dp.
        new_dp = dp[:]
        for cost, happy in choices:
            # For each possible capacity, update if adding this group is possible.
            # We iterate backwards to ensure we do not reuse an item type more than once.
            for j in range(W - cost, -1, -1):
                if dp[j] != -10**18:
                    new_dp[j + cost] = max(new_dp[j + cost], dp[j] + happy)
        dp = new_dp
    
    # The answer is the maximum happiness over all weights within the capacity.
    print(max(dp))
    
if __name__ == "__main__":
    main()