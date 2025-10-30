import sys

def solve():
    N, X = map(int, sys.stdin.readline().split())

    foods_by_type = [[] for _ in range(3)]
    for _ in range(N):
        V, A, C = map(int, sys.stdin.readline().split())
        # V is 1-indexed, convert to 0-indexed for lists
        foods_by_type[V - 1].append((A, C)) # (Amount, Cost)

    # dp[v_idx][cost] = max_vitamin_amount for vitamin type v_idx 
    #                   achievable with exactly 'cost' calories using only foods of this type.
    # Initialize with 0, meaning 0 vitamin amount for 0 cost, or if no items taken.
    dp = [[0] * (X + 1) for _ in range(3)]

    for v_idx in range(3):
        # Standard 0/1 knapsack variant for each vitamin type
        # Items are foods of this type. "Weight" is C_i, "Value" is A_i.
        # We want to maximize total A_i for a given total C_i.
        for amount, cost_food in foods_by_type[v_idx]:
            # Iterate cost downwards to ensure each food is considered at most once
            for c in range(X, cost_food - 1, -1):
                # Option 1: Don't include this food, vitamin amount is dp[v_idx][c] (already computed from other foods)
                # Option 2: Include this food. Vitamin amount is dp[v_idx][c - cost_food] + amount
                dp[v_idx][c] = max(dp[v_idx][c], dp[v_idx][c - cost_food] + amount)
    
    best_m = 0 # Stores the maximum m found to be achievable
    
    # Binary search for the maximum m
    # Smallest possible m is 0.
    # Largest possible m could be high (e.g., 10^9, if A_i are large and C_i are small).
    low = 0
    high = 10**9 + 7 # A sufficiently large upper bound for m (e.g. max_X * max_A_i or max_N * max_A_i)

    # check(k_target):
    # Returns True if it's possible to get at least k_target units of each vitamin
    # type (0, 1, 2) such that total calories consumed for these vitamins is <= X.
    # Returns False otherwise.
    def check(k_target):
        if k_target == 0: # 0 units of all vitamins is always achievable with 0 cost.
            return True

        total_min_cost_for_k_target = 0
        for v_idx in range(3): # For each vitamin type
            current_min_cost_for_v = float('inf')
            found_possible_cost_for_this_vitamin = False
            # Find minimum cost 'c' to get at least k_target units of this vitamin
            for c_loopvar in range(X + 1): # Cost can be from 0 to X
                if dp[v_idx][c_loopvar] >= k_target:
                    current_min_cost_for_v = c_loopvar
                    found_possible_cost_for_this_vitamin = True
                    break # Found smallest c for this vitamin type, move to next
            
            if not found_possible_cost_for_this_vitamin:
                # If for any vitamin type we cannot achieve k_target units even spending up to X calories
                # on foods of this type, then k_target is not achievable overall.
                return False 
            
            total_min_cost_for_k_target += current_min_cost_for_v
            
            # Optimization: if accumulated cost already exceeds X, no need to check further
            if total_min_cost_for_k_target > X:
                return False

        # If loop completes, it means we found costs for all 3 vitamin types.
        # The final check is implicitly done by the optimization above, 
        # but can be stated explicitly: return total_min_cost_for_k_target <= X
        return True


    while low <= high:
        mid = low + (high - low) // 2
        # mid == 0 is a special case to ensure we don't test check(0) if mid is 0 from calculation,
        # as check(0) is always true. It's better to let check handle k_target=0.
        # The condition "if mid == 0: ..." in thought process was simplified to below.
        # best_m is initialized to 0. If check(mid) is true for mid > 0, best_m gets updated.
        # If no mid > 0 passes, best_m remains 0.

        if check(mid):
            best_m = mid # mid is achievable, try for a larger m
            low = mid + 1
        else:
            high = mid - 1 # mid is not achievable, try smaller m
            
    print(best_m)

solve()