import math

def solve():
    N = int(input())
    P_original = list(map(int, input().split()))

    # dp[i][k] stores the maximum numerator sum S_k for a selection of k contests,
    # where P_original[i] (0-indexed) is the k-th chosen contest.
    # i ranges from 0 to N-1.
    # k ranges from 1 to N.
    # dp table: N rows, (N+1) columns (column 0 is unused for k)
    dp = [[0.0] * (N + 1) for _ in range(N)]

    # D_arr[k] stores D_k = sum_{j=0}^{k-1} (0.9)^j
    # D_arr[0] is unused. D_arr[1]=1. D_arr[k] = 1 + 0.9 * D_arr[k-1] for k>1.
    D_arr = [0.0] * (N + 1)
    if N >= 1: # N is guaranteed to be >= 1 by constraints
        D_arr[1] = 1.0
    for k_val in range(2, N + 1):
        D_arr[k_val] = 1.0 + 0.9 * D_arr[k_val - 1]

    # C_arr[k] stores C_k = 1200 / sqrt(k)
    # C_arr[0] is unused.
    C_arr = [0.0] * (N + 1)
    for k_val in range(1, N + 1):
        C_arr[k_val] = 1200.0 / math.sqrt(k_val)

    max_R = -float('inf') # Max rating found so far

    # Base case: k_count = 1 (choosing exactly one contest)
    # P_original[i] is the 1st (and only) chosen contest.
    for i in range(N):
        dp[i][1] = float(P_original[i])
        rating = dp[i][1] / D_arr[1] - C_arr[1]
        if rating > max_R:
            max_R = rating
            
    # DP for k_count from 2 to N
    for k_count in range(2, N + 1):
        # P_original[i] is the k_count-th chosen contest.
        # Its original index i must be at least k_count-1.
        # (e.g., if k_count=2, P_original[i] is 2nd chosen, i must be at least 1).
        
        # The (k_count-1)-th chosen contest, P_original[m], must have m < i.
        # Also, P_original[m] is (k_count-1)-th chosen, so m must be at least k_count-2
        # (original index of (k_count-1)-th item is at least (k_count-1)-1 ).
        # Thus, m ranges from k_count-2 to i-1.
        # We need max_val = max(dp[m][k_count-1] for m in [k_count-2, i-1]).
        
        # current_max_prev_dp_val will track this maximum iteratively.
        # Initialize for the first possible i, which is i = k_count-1.
        # For i = k_count-1, m must be k_count-2.
        # So, the value needed is dp[k_count-2][k_count-1].
        current_max_prev_dp_val = dp[k_count - 2][k_count - 1]
        
        # Iterate i from k_count-1 (earliest possible original index for k_count-th item) up to N-1.
        for i in range(k_count - 1, N):
            if i > k_count - 1:
                # We are now considering P_original[i] as k_count-th item.
                # The previous item P_original[i-1] could have been (k_count-1)-th item.
                # So, update current_max_prev_dp_val to include dp[i-1][k_count-1].
                # current_max_prev_dp_val now holds max(dp[m][k_count-1] for m in [k_count-2, i-1]).
                current_max_prev_dp_val = max(current_max_prev_dp_val, dp[i - 1][k_count - 1])
            
            dp[i][k_count] = float(P_original[i]) + 0.9 * current_max_prev_dp_val
            
            rating = dp[i][k_count] / D_arr[k_count] - C_arr[k_count]
            if rating > max_R:
                max_R = rating
                
    print(f"{max_R:.17f}")

solve()