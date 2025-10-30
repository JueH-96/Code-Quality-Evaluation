import sys

def solve():
    N, X = map(int, sys.stdin.readline().split())

    # Group foods by vitamin type for easier processing
    foods_by_vitamin = {1: [], 2: [], 3: []}
    for _ in range(N):
        V, A, C = map(int, sys.stdin.readline().split())
        foods_by_vitamin[V].append((A, C))

    # Initialize DP tables for each vitamin type.
    # dp_v[c] will store the maximum vitamin amount achievable with exactly 'c' calories
    # for foods of type 'v'.
    # -1 indicates an unreachable state for that calorie count.
    dp1 = [-1] * (X + 1)
    dp2 = [-1] * (X + 1)
    dp3 = [-1] * (X + 1)
    
    # Base case: 0 calories yields 0 vitamin amount
    dp1[0] = 0
    dp2[0] = 0
    dp3[0] = 0

    # Populate dp tables using a 0/1 knapsack-like approach
    # For each food (A_i, C_i), iterate through calorie capacities downwards
    # to ensure each food item is used at most once.

    # For Vitamin 1
    for A, C in foods_by_vitamin[1]:
        for c in range(X, C - 1, -1): # Iterate downwards from X to C
            if dp1[c - C] != -1: # If the previous state (c - C calories) was reachable
                dp1[c] = max(dp1[c], dp1[c - C] + A)

    # For Vitamin 2
    for A, C in foods_by_vitamin[2]:
        for c in range(X, C - 1, -1):
            if dp2[c - C] != -1:
                dp2[c] = max(dp2[c], dp2[c - C] + A)

    # For Vitamin 3
    for A, C in foods_by_vitamin[3]:
        for c in range(X, C - 1, -1):
            if dp3[c - C] != -1:
                dp3[c] = max(dp3[c], dp3[c - C] + A)

    # Helper function to check if a minimum intake 'k' is achievable
    def check(k):
        # Find the minimum calories needed for each vitamin type to reach at least 'k'
        cost1_for_k = X + 1 # Initialize with a value > X to indicate impossible
        for c1 in range(X + 1):
            if dp1[c1] >= k:
                cost1_for_k = c1
                break # Found the minimum calorie cost, no need to check higher 'c1'

        cost2_for_k = X + 1
        for c2 in range(X + 1):
            if dp2[c2] >= k:
                cost2_for_k = c2
                break

        cost3_for_k = X + 1
        for c3 in range(X + 1):
            if dp3[c3] >= k:
                cost3_for_k = c3
                break
        
        # If the sum of minimum costs for all three vitamins does not exceed X,
        # then 'k' is achievable.
        return cost1_for_k + cost2_for_k + cost3_for_k <= X

    # Binary search for the maximum possible 'k'
    low = 0
    # The upper bound for 'k' can be at most N * max(A_i).
    # max(A_i) is 2 * 10^5, N is 5000, so 5000 * 2 * 10^5 = 10^9.
    # Add 1 to be inclusive for potential max value
    high = 2 * 10**5 * N + 1 
    ans = 0 # Stores the maximum 'k' found so far

    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans = mid      # 'mid' is achievable, try for a higher minimum
            low = mid + 1
        else:
            high = mid - 1 # 'mid' is not achievable, reduce the minimum target
            
    sys.stdout.write(str(ans) + "
")

# Call the solve function to run the program
solve()