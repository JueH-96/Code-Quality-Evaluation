import sys

def solve():
    """
    Reads input, solves the problem, and prints the answer.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        # Read N and X from the first line of stdin
        line = input()
        if not line:
            return
        N, X = map(int, line.split())

        # Group foods by their vitamin type
        foods_by_type = {1: [], 2: [], 3: []}
        for _ in range(N):
            line = input()
            if not line:
                break
            v, a, c = map(int, line.split())
            foods_by_type[v].append((a, c))

    except (IOError, ValueError):
        return

    def compute_knapsack_dp(foods, max_cost):
        """
        Computes the maximum vitamin amount for each possible cost from 0 to max_cost.
        This is a standard 0/1 knapsack problem formulation.
        - dp[c]: max vitamin amount achievable with cost c.
        - foods: a list of (vitamin_amount, cost) tuples.
        - max_cost: the maximum total cost (X).
        """
        dp = [0] * (max_cost + 1)
        for vitamin_amount, cost in foods:
            for c in range(max_cost, cost - 1, -1):
                dp[c] = max(dp[c], dp[c - cost] + vitamin_amount)
        return dp

    # Pre-compute DP tables for each vitamin type.
    dp1 = compute_knapsack_dp(foods_by_type[1], X)
    dp2 = compute_knapsack_dp(foods_by_type[2], X)
    dp3 = compute_knapsack_dp(foods_by_type[3], X)

    def can_achieve(k):
        """
        Check function for the binary search.
        Determines if it's possible to get at least 'k' units of each vitamin
        without exceeding the total calorie limit X.
        """
        # For each vitamin type, find the minimum cost to achieve at least 'k' units.
        
        min_cost1 = float('inf')
        for c, v in enumerate(dp1):
            if v >= k:
                min_cost1 = c
                break
        if min_cost1 == float('inf'):
            return False

        min_cost2 = float('inf')
        for c, v in enumerate(dp2):
            if v >= k:
                min_cost2 = c
                break
        if min_cost2 == float('inf'):
            return False
        
        min_cost3 = float('inf')
        for c, v in enumerate(dp3):
            if v >= k:
                min_cost3 = c
                break
        if min_cost3 == float('inf'):
            return False
            
        # The total cost is the sum of minimum costs for each type.
        return min_cost1 + min_cost2 + min_cost3 <= X

    # Binary search for the maximum possible value of 'k'.
    ans = 0
    # A safe upper bound for k.
    low, high = 0, 10**9 + 7

    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            # 0 is always achievable and handled by ans=0 initialization.
            # This avoids an unnecessary check.
            low = mid + 1
            continue
        
        if can_achieve(mid):
            # If 'mid' is achievable, we store it and try for a larger value.
            ans = mid
            low = mid + 1
        else:
            # If 'mid' is not achievable, we must try a smaller value.
            high = mid - 1

    print(ans)

solve()