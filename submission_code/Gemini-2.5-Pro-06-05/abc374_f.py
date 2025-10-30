import sys

def solve():
    """
    Solves the KEYENCE shipping problem using dynamic programming.
    """
    
    # Read input from stdin
    try:
        line1 = sys.stdin.readline()
        if not line1: return
        N, K, X = map(int, line1.split())
        
        line2 = sys.stdin.readline()
        if not line2: return
        T = list(map(int, line2.split()))
    except (IOError, ValueError):
        return

    # Use a very large number to represent infinity.
    infinity = float('inf')

    # dp[m][i] stores a tuple: (minimum_dissatisfaction, last_shipping_day)
    # for shipping the first i items using exactly m shipments.
    # We use (N+1) x (N+1) size for 1-based indexing of m and i.
    dp = [[(infinity, infinity) for _ in range(N + 1)] for _ in range(N + 1)]

    # Precompute prefix sums of T for O(1) calculation of the sum of T's in a batch.
    # P[i] stores the sum of the first i elements of T (i.e., T[0] to T[i-1]).
    P = [0] * (N + 1)
    for i in range(N):
        P[i + 1] = P[i] + T[i]

    # Iterate through the number of shipments, m
    for m in range(1, N + 1):
        # Iterate through the total number of items shipped, i
        for i in range(m, N + 1):
            # Base case: m = 1. Shipping items 1..i in a single shipment.
            if m == 1:
                # This is only possible if the number of items, i, is at most K.
                if i <= K:
                    # To minimize dissatisfaction, ship on the latest placement day of the batch.
                    s_day = T[i - 1]
                    # Dissatisfaction = sum_{k=1 to i} (s_day - T_k)
                    #                 = i * s_day - sum_{k=1 to i} T_k
                    # In 0-based T, T_k is T[k-1]. The sum is P[i].
                    diss = i * s_day - P[i]
                    dp[m][i] = (diss, s_day)
            
            # Recursive step: m > 1.
            # The last shipment contains items from index j to i.
            # The previous m-1 shipments handled items 1 to j-1.
            else:
                # Determine the valid range for j, the 1-based start index of the last batch.
                # Constraint 1: Batch size (i-j+1) <= K  =>  j >= i - K + 1
                # Constraint 2: First j-1 items shipped in m-1 groups => j-1 >= m-1 => j >= m
                start_j = max(m, i - K + 1)
                for j in range(start_j, i + 1):
                    prev_diss, prev_s_day = dp[m - 1][j - 1]

                    # If the previous state is unreachable, this path is not possible.
                    if prev_diss == infinity:
                        continue
                    
                    # The shipping day for the current batch is the earliest possible day,
                    # which is the max of the placement day constraint and the time-gap constraint.
                    s_day = max(T[i - 1], prev_s_day + X)
                    
                    # Calculate dissatisfaction for the current batch {j..i}.
                    num_items_batch = i - j + 1
                    sum_T_batch = P[i] - P[j - 1]
                    diss_batch = num_items_batch * s_day - sum_T_batch
                    
                    total_diss = prev_diss + diss_batch

                    # Update dp[m][i] if we found a better schedule.
                    # A schedule is "better" if it has lower total dissatisfaction.
                    # For a tie in dissatisfaction, the one with an earlier final shipping day
                    # is preferred, as it may lead to better outcomes for subsequent shipments.
                    current_best_diss, current_best_s_day = dp[m][i]
                    if total_diss < current_best_diss:
                        dp[m][i] = (total_diss, s_day)
                    elif total_diss == current_best_diss and s_day < current_best_s_day:
                        dp[m][i] = (total_diss, s_day)

    # The final answer is the minimum dissatisfaction for shipping all N items,
    # considering all possible numbers of shipments (from 1 to N).
    min_total_diss = infinity
    for m in range(1, N + 1):
        min_total_diss = min(min_total_diss, dp[m][N][0])

    # Print the result to stdout.
    print(min_total_diss)

solve()