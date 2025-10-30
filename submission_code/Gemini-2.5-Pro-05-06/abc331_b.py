import math

def solve():
    N, S, M, L = map(int, input().split())

    # The maximum index in the DP table we need to consider is N_max + max_pack_size - 1.
    # N_max = 100, max_pack_size = 12. So, 100 + 12 - 1 = 111.
    DP_TABLE_LIMIT_IDX = 111 
    
    # dp[i] will store the minimum cost to get exactly i eggs.
    # Initialize dp array. Size is DP_TABLE_LIMIT_IDX + 1 for indices 0 to DP_TABLE_LIMIT_IDX.
    dp = [math.inf] * (DP_TABLE_LIMIT_IDX + 1)
    dp[0] = 0  # Cost to get 0 eggs is 0.

    pack_options = [
        {'size': 6, 'cost': S},
        {'size': 8, 'cost': M},
        {'size': 12, 'cost': L}
    ]

    # Fill the DP table
    for i in range(1, DP_TABLE_LIMIT_IDX + 1):
        for option in pack_options:
            size = option['size']
            cost = option['cost']
            if i - size >= 0:
                # If dp[i-size] is math.inf, it means i-size eggs cannot be formed (or is part of an unreachably expensive path).
                # math.inf + cost is still math.inf, so min comparison works correctly.
                # Explicitly checking `if dp[i-size] != math.inf:` is fine but not strictly necessary.
                dp[i] = min(dp[i], dp[i-size] + cost)
    
    min_total_cost = math.inf
    # We need at least N eggs.
    # Iterate from N up to DP_TABLE_LIMIT_IDX to find the minimum cost.
    for i in range(N, DP_TABLE_LIMIT_IDX + 1):
        min_total_cost = min(min_total_cost, dp[i])
        
    print(int(min_total_cost))

# This structure is for competitive programming platforms.
# The solve() function will be called to run the solution.
if __name__ == '__main__':
    solve()