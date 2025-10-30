n, k, x = map(int, input().split())
t = list(map(int, input().split()))

# Compute prefix sums for efficient sum calculation
prefix_sum = [0] * (n + 1)
for i in range(n):
    prefix_sum[i + 1] = prefix_sum[i] + t[i]

# Initialize DP where each entry is a dictionary of {earliest_day: dissatisfaction}
dp = [{} for _ in range(n + 1)]
dp[0][0] = 0  # Starting at 0 orders processed, earliest day 0, dissatisfaction 0

for i in range(n + 1):
    current_dict = dp[i]
    if not current_dict:
        continue
    # Iterate through a snapshot of current_dict to avoid runtime errors
    for e in list(current_dict.keys()):
        current_cost = current_dict[e]
        # Try all possible group sizes m from 1 to k
        for m in range(1, k + 1):
            j = i + m - 1
            if j >= n:
                continue  # This group exceeds available orders
            max_t = t[j]
            s = max(e, max_t)  # Shipment day for this group
            sum_t = prefix_sum[j + 1] - prefix_sum[i]
            added_dissatisfaction = m * s - sum_t
            new_cost = current_cost + added_dissatisfaction
            new_earliest_day = s + x
            next_i = i + m
            if next_i > n:
                continue  # Skip if next_i exceeds the total orders
            # Update the next state's dictionary
            if new_earliest_day in dp[next_i]:
                if new_cost < dp[next_i][new_earliest_day]:
                    dp[next_i][new_earliest_day] = new_cost
            else:
                dp[next_i][new_earliest_day] = new_cost

# The answer is the minimum dissatisfaction in dp[n]
print(min(dp[n].values()) if dp[n] else 0)