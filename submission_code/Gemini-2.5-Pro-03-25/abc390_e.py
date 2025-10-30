# YOUR CODE HERE
import sys

def solve():
    # Read N (number of foods) and X (max total calories)
    # N: number of foods
    # X: maximum total calorie consumption allowed
    N, X = map(int, sys.stdin.readline().split())
    
    # Initialize lists to store foods for each vitamin type (1, 2, 3)
    # foods[j] will contain a list of tuples (A_i, C_i) for foods providing vitamin j
    # A_i: vitamin amount
    # C_i: calories (cost)
    foods = [[] for _ in range(4)] # Index 0 is unused, use indices 1, 2, 3 for convenience
    
    # Read food data for N foods
    for _ in range(N):
        V, A, C = map(int, sys.stdin.readline().split())
        # V: vitamin type (1, 2, or 3)
        # A: vitamin amount
        # C: calories
        
        # We only need to consider foods whose calorie cost C is not greater than the total budget X.
        # The problem constraints state 1 <= C_i <= X, so this check is technically redundant
        # based on the constraints, but good practice for potentially varied inputs.
        if C <= X:
             foods[V].append((A, C)) 

    # Initialize DP tables for each vitamin type
    # DP[j][c] will store the maximum amount of vitamin j achievable using exactly c calories.
    # Dimensions: 4 vitamin types (index 0 unused) x (X+1) possible costs (from 0 to X)
    # Initialize all states to -1, indicating "unreachable" or "not yet computed".
    DP = [[-1] * (X + 1) for _ in range(4)]

    # Compute the DP tables for each vitamin type using a standard 0/1 knapsack dynamic programming approach.
    for j in range(1, 4): # Iterate through vitamin types 1, 2, 3
        # Base case: It's always possible to achieve 0 vitamin amount with 0 calories.
        DP[j][0] = 0 
        
        # Iterate through each food item that provides vitamin j
        for A_i, C_i in foods[j]:
            # Update the DP table based on the possibility of including this food item.
            # We iterate the cost 'c' downwards from X to C_i. This ensures that each food item
            # is considered at most once for any particular total cost 'c', fulfilling the 0/1 property
            # (i.e., we either take the item or not, not multiple times).
            for c in range(X, C_i - 1, -1):
                # Check if the state (c - C_i) is reachable (i.e., DP[j][c - C_i] is not -1).
                # This means it's possible to achieve some vitamin amount with cost c - C_i.
                if DP[j][c - C_i] != -1:
                    # If reachable, calculate the potential new maximum vitamin amount for cost 'c'
                    # by adding the current food's vitamin amount A_i to the amount achieved at cost c - C_i.
                    potential_new_A = DP[j][c - C_i] + A_i
                    
                    # Update DP[j][c] if this new path (including the current food) yields a higher
                    # vitamin amount than what's currently recorded for cost 'c', or if DP[j][c] 
                    # was previously unreachable (-1).
                    DP[j][c] = max(DP[j][c], potential_new_A)

    # Compute prefix maximums for the DP table of vitamin 3.
    # MaxDP3[k] will store the maximum vitamin 3 amount achievable with a total calorie cost of AT MOST k.
    # This helps optimize the final step where we combine results for the three vitamins.
    MaxDP3 = [-1] * (X + 1)
    current_max_v3 = -1 # Track the running maximum vitamin 3 amount found so far. Initialized to -1.
    for k in range(X + 1): # Iterate through all possible costs from 0 to X
        # Update the running maximum by comparing it with the vitamin amount achievable at EXACTLY cost k (DP[3][k]).
        # We take the maximum of the current running max and DP[3][k]. Note that max(-1, x) = x for x >= -1.
        current_max_v3 = max(current_max_v3, DP[3][k])
        
        # Store this running maximum in MaxDP3[k]. 
        # MaxDP3[k] now holds the max vitamin 3 amount for any cost c <= k.
        MaxDP3[k] = current_max_v3

    # Initialize the variable to store the overall maximum possible K value found.
    # K represents the minimum intake among the three vitamins. We want to maximize K.
    max_K = 0

    # Iterate through all possible calorie allocations (c1, c2, c3) such that c1 + c2 + c3 <= X.
    # We do this efficiently by iterating through all possible costs c1 for vitamin 1 and c2 for vitamin 2.
    # The remaining budget automatically determines the maximum possible cost c3 for vitamin 3.
    
    # Iterate through possible calories c1 allocated to vitamin 1 foods (from 0 to X)
    for c1 in range(X + 1):
        # If cost c1 is not achievable for vitamin 1 (DP[1][c1] is -1), skip this allocation.
        if DP[1][c1] == -1:
            continue
        
        # Get the maximum vitamin 1 amount achievable with exactly c1 calories.
        v1 = DP[1][c1]
        
        # Calculate the remaining calorie budget after spending c1.
        remaining_X_after_c1 = X - c1
        
        # Iterate through possible calories c2 allocated to vitamin 2 foods.
        # c2 can range from 0 up to the remaining budget `remaining_X_after_c1`.
        for c2 in range(remaining_X_after_c1 + 1):
            # If cost c2 is not achievable for vitamin 2 (DP[2][c2] is -1), skip this allocation.
            if DP[2][c2] == -1:
                continue

            # Get the maximum vitamin 2 amount achievable with exactly c2 calories.
            v2 = DP[2][c2]
            
            # Calculate the remaining calorie budget for vitamin 3 foods after spending c1 and c2.
            remaining_X_after_c1_c2 = remaining_X_after_c1 - c2
            # This budget is guaranteed to be non-negative because the loop ensures c2 <= remaining_X_after_c1.
            
            # Find the maximum vitamin 3 amount achievable with a calorie cost AT MOST `remaining_X_after_c1_c2`.
            # We use the precomputed prefix maximum table MaxDP3 for efficient lookup.
            v3 = MaxDP3[remaining_X_after_c1_c2]

            # Check if it's possible to obtain a non-negative amount of vitamin 3 within the budget.
            # Since DP[3][0] = 0, MaxDP3[k] will be >= 0 for all k >= 0.
            # Therefore, v3 will always be >= 0 if remaining_X_after_c1_c2 >= 0.
            # The check `if v3 != -1` essentially verifies that a valid vitamin 3 amount was found
            # (which will always be the case here since 0 is always possible with 0 cost).
            if v3 != -1: 
                # Calculate the minimum vitamin intake among the three types for this specific calorie allocation (c1, c2, <= remaining).
                current_K = min(v1, v2, v3)
                
                # Update the overall maximum K found so far across all valid allocations.
                max_K = max(max_K, current_K)

    # Print the final result, which is the maximum possible value for the minimum vitamin intake.
    print(max_K)

# Call the solve function to execute the logic.
solve()