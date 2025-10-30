import sys

# Function to compute the maximum vitamin units achievable for each possible calorie cost
# using a given list of foods (A, C) for a specific vitamin type, within budget X.
# Returns a DP table where dp[c] is the maximum vitamin for exactly c calories.
# This uses a 0/1 knapsack-like DP because each food is a unique item.
def compute_max_vitamin_dp(foods, X):
    # dp[c] will store the maximum vitamin units achievable with exactly c calories.
    # Initialize dp table. Use -1 to represent unreachable states.
    dp = [-1] * (X + 1)
    dp[0] = 0 # 0 calories can achieve 0 vitamin

    # Iterate through each food (amount, cost) of this vitamin type
    for amount, cost in foods:
        # Iterate through calorie budget from X down to cost
        # This ensures each food is used at most once (0/1 knapsack style)
        for c in range(X, cost - 1, -1):
            # If the state c - cost is reachable
            if dp[c - cost] != -1:
                # Update dp[c] with the maximum vitamin amount
                dp[c] = max(dp[c], dp[c - cost] + amount)

    return dp

# Function to compute the minimum cost to get at least k vitamin units
# using a given DP table (from compute_max_vitamin_dp) and a calorie budget X.
def min_cost_from_dp(dp, k, X):
    # Find the minimum calorie cost to achieve at least k vitamin units
    min_c = X + 1 # Initialize with a value greater than max possible cost X
    for c in range(X + 1):
        if dp[c] >= k:
            min_c = c
            break # Found the minimum cost

    return min_c


# Function to check if a minimum intake of k is feasible within calorie budget X
def is_feasible(k, foods_by_vitamin, X):
    # If k is 0, it's always feasible (take no foods, cost 0, min vitamin 0).
    if k == 0:
        return True

    # Compute the max vitamin DP table for each vitamin type with full budget X
    dp1 = compute_max_vitamin_dp(foods_by_vitamin.get(1, []), X)
    dp2 = compute_max_vitamin_dp(foods_by_vitamin.get(2, []), X)
    dp3 = compute_max_vitamin_dp(foods_by_vitamin.get(3, []), X)

    # Find the minimum cost to achieve at least k units for each vitamin type
    # If a vitamin type has no foods or cannot reach k within budget X, min_cost will be X + 1
    cost1 = min_cost_from_dp(dp1, k, X)
    cost2 = min_cost_from_dp(dp2, k, X)
    cost3 = min_cost_from_dp(dp3, k, X)

    # Check if the sum of the minimum costs is within the total budget X
    # If any cost_v is X+1, the sum cost1+cost2+cost3 will be >= X+1, which correctly fails the <= X check.
    return cost1 + cost2 + cost3 <= X


def solve():
    # Read input N and X
    N, X = map(int, sys.stdin.readline().split())

    # Separate foods by vitamin type
    foods_by_vitamin = {1: [], 2: [], 3: []}
    for _ in range(N):
        V, A, C = map(int, sys.stdin.readline().split())
        foods_by_vitamin[V].append((A, C))

    # Calculate the maximum possible vitamin amount achievable for each type within budget X
    # This is used to determine the upper bound for binary search on k.
    # Compute DP table for each vitamin type with full budget X
    dp1_full = compute_max_vitamin_dp(foods_by_vitamin.get(1, []), X)
    dp2_full = compute_max_vitamin_dp(foods_by_vitamin.get(2, []), X)
    dp3_full = compute_max_vitamin_dp(foods_by_vitamin.get(3, []), X)

    # Find the maximum vitamin amount achieved for each type with budget up to X
    # This is the maximum value in the DP table. If no foods of a type exist, the dp is [-1]*X + [0], max is 0.
    # If foods exist, max is at least 0.
    max_v1 = max(dp1_full)
    max_v2 = max(dp2_full)
    max_v3 = max(dp3_full)

    # The maximum possible value for the minimum intake k is bounded
    # by the minimum of the maximum achievable amounts for each vitamin type.
    # Add 1 for binary search upper bound (search range [low, high)).
    high = min(max_v1, max_v2, max_v3) + 1

    # Binary search for the maximum feasible minimum vitamin intake k
    low = 0
    ans = 0 # k = 0 is always feasible

    # Search space [low, high)
    while low < high:
        mid = (low + high) // 2
        if is_feasible(mid, foods_by_vitamin, X):
            # mid is feasible, try a higher value
            ans = mid
            low = mid + 1
        else:
            # mid is not feasible, need to aim lower
            high = mid # Search in [low, mid)

    print(ans)

solve()