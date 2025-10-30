import sys

def solve():
    N, K, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix sums for T to get sum(T_k) in O(1)
    # pref_T[i] will store sum(T[0]...T[i-1])
    pref_T = [0] * (N + 1)
    for i in range(N):
        pref_T[i+1] = pref_T[i] + T[i]

    # dp[i] will store the minimum total dissatisfaction for shipping orders 0 to i-1
    # last_ship_day[i] will store the actual day the (i-1)-th order was shipped 
    # when dp[i] minimum dissatisfaction is achieved. This helps achieve minimum dissatisfaction
    # for future orders, as an earlier last_ship_day allows an earlier next_available_shipment_day.
    dp = [float('inf')] * (N + 1)
    last_ship_day = [float('inf')] * (N + 1)

    # Base case: 0 orders shipped, 0 dissatisfaction.
    # last_ship_day[0] represents a fictitious state before any shipment.
    # A value of 1 allows the very first shipment to be on Day 1 (if T[0] is 1)
    # or later (max(1, T[0])).
    dp[0] = 0
    last_ship_day[0] = 1 

    # Iterate through each possible number of shipped orders, from 1 to N
    # 'i' represents the number of orders shipped (orders 0 to i-1)
    for i in range(1, N + 1):
        # Iterate through possible starting indices 'j' for the current batch
        # This means orders T[j]...T[i-1] are shipped in this batch.
        # The previous 'j' orders (0 to j-1) must have been shipped already.
        # Number of orders in this batch is (i - j).
        # This number must be between 1 and K. So, j can range from max(0, i - K) to i - 1.
        for j in range(max(0, i - K), i):
            prev_total_diss = dp[j]
            
            # If dp[j] is still infinity, it means this path is not reachable.
            # This check is crucial for handling cases where previous states might not be valid.
            if prev_total_diss == float('inf'):
                continue

            prev_LSD = last_ship_day[j]
            
            # Calculate the earliest time the current batch can be shipped.
            # If j == 0, this is the very first shipment, so it can be shipped from Day 1.
            # Otherwise, it must respect the cooldown period from the previous shipment (at prev_LSD).
            current_available_time = 1 if j == 0 else prev_LSD + X
            
            # The actual shipping day for this batch must be at least current_available_time
            # and also at least T[i-1] (the placement day of the latest order in this batch).
            actual_ship_day = max(current_available_time, T[i-1])
            
            # Calculate dissatisfaction for the current batch
            num_orders_in_batch = i - j
            sum_T_in_batch = pref_T[i] - pref_T[j] # Sum of T values for orders T[j]...T[i-1]
            current_batch_diss = (num_orders_in_batch * actual_ship_day) - sum_T_in_batch
            
            # Calculate new total dissatisfaction
            new_total_diss = prev_total_diss + current_batch_diss
            
            # Update dp[i] if a better path is found
            if new_total_diss < dp[i]:
                dp[i] = new_total_diss
                last_ship_day[i] = actual_ship_day
            elif new_total_diss == dp[i]:
                # If total dissatisfaction is the same, choose the path that results in an earlier 
                # last_ship_day. This is beneficial because an earlier last_ship_day can lead 
                # to an earlier next_available_shipment_day for subsequent batches, potentially 
                # reducing future dissatisfaction.
                last_ship_day[i] = min(last_ship_day[i], actual_ship_day)

    # The minimum total dissatisfaction for all N orders is stored in dp[N]
    print(dp[N])

solve()