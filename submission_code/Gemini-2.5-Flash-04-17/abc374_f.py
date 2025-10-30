import sys

# Read input
line1 = sys.stdin.readline().split()
N = int(line1[0])
K = int(line1[1])
X = int(line1[2])
T_str = sys.stdin.readline().split()
T = [int(t) for t in T_str]

# Calculate prefix sums of T
# SumT[i] = sum(T[0]...T[i-1])
SumT = [0] * (N + 1)
for i in range(N):
    SumT[i+1] = SumT[i] + T[i]

# dp[i] = (min_total_dissatisfaction_for_first_i_orders, day_the_batch_containing_order_i-1_was_shipped)
# dp has size N+1. dp[i] refers to the state after processing the first i orders (indices 0 to i-1).

# Use large integers for infinity. Python's arbitrary precision integers handle large values.
# Max possible total dissatisfaction approx N * (T_N + N*X - T_1) = 100 * (10^12 + 100*10^9 - 1) approx 1.1 * 10^14
INF_DISS = 2 * 10**14 # Sufficiently large
# Max possible ship day approx T_N + N*X = 10^12 + 100*10^9 approx 1.1 * 10^12
INF_DAY = 2 * 10**12 # Sufficiently large

dp = [(INF_DISS, INF_DAY)] * (N + 1)

# dp[0] state: 0 orders processed, 0 total dissatisfaction.
# The second element is the day the *previous* batch was shipped.
# For dp[0], representing the state before the very first shipment,
# this day should be conceptually very early, so that prev_ship_day + X
# does not constrain the first real shipment unless T[i-1] is very small.
# We need prev_ship_day + X <= T[i-1]. Smallest T[i-1] for i>=1 is T[0] >= 1.
# We need prev_ship_day <= T[0] - X. Using a value << min(T) - X is safe.
# T_i >= 1, X >= 1. min(T) - X can be 1 - 10^9.
# A value like -2 * 10**12 should be safely smaller than any T[0] - X.
dp[0] = (0, -2 * 10**12) 

# For i from 1 to N:
# dp[i] considers shipping orders 0 to i-1, ending with order i-1 in the last batch.
for i in range(1, N + 1):
    # The last batch contains orders from index j to i-1.
    # The previous state processed orders 0 to j-1, stored in dp[j].
    # Number of orders in last batch = (i-1) - j + 1 = i - j.
    # This size must be between 1 and K.
    # 1 <= i - j <= K  => i - K <= j <= i - 1.
    # The previous state index j must be at least 0 (representing processing 0 orders).
    # So, j ranges from max(0, i - K), inclusive, up to i, exclusive.
    
    for j in range(max(0, i - K), i):
        prev_diss, prev_ship_day = dp[j]

        # If the previous state is unreachable, skip it
        if prev_diss >= INF_DISS: # Use >= for checking infinity sentinel
            continue

        # The current batch contains orders T[j]...T[i-1].
        # The latest placement day in this batch is T[i-1].
        
        # The earliest day this batch can be shipped is the maximum of:
        # 1. The latest placement day in the batch (T[i-1])
        # 2. X days after the previous shipment (prev_ship_day + X)
        # For j=0, prev_ship_day = dp[0][1] = -2 * 10**12. prev_ship_day + X is small.
        # So current_ship_day will be max(T[i-1], small_number). Since T[i-1] >= 1,
        # max(T[i-1], small_number) = T[i-1] if T[i-1] >= small_number, which is true.
        # This effectively models the first shipment not being constrained by a previous X rule relative to t=0,
        # unless T[i-1] is so small that X dominates the max(), which can only happen if X is very large and T[i-1] is negative or very small relative to X.
        # Given T_i >= 1, max(T[i-1], prev_ship_day + X) correctly applies the constraint from the previous real or hypothetical shipment.
        current_ship_day = max(T[i-1], prev_ship_day + X)

        # Batch dissatisfaction: sum_{k=j}^{i-1} (current_ship_day - T[k])
        # = (Number of items in batch) * current_ship_day - (Sum of T values in batch)
        # Number of items = i - j
        # Sum of T values = SumT[i] - SumT[j]
        batch_diss = (i - j) * current_ship_day - (SumT[i] - SumT[j])
        
        # Total dissatisfaction for the first i orders
        total_diss = prev_diss + batch_diss

        # Update dp[i]
        if total_diss < dp[i][0]:
            dp[i] = (total_diss, current_ship_day)
        elif total_diss == dp[i][0]:
            # If total dissatisfaction is tied, prefer the schedule that results in an earlier shipment day for the last batch.
            # This heuristic aims to potentially allow subsequent shipments sooner, which can help minimize future dissatisfaction.
            if current_ship_day < dp[i][1]:
                dp[i] = (total_diss, current_ship_day)

# The minimum total dissatisfaction for all N orders is stored in dp[N][0]
print(dp[N][0])