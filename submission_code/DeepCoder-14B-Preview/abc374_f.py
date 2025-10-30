n, k, x = map(int, input().split())
t = list(map(int, input().split()))
prefix_sum = [0] * (n + 1)
for i in range(n):
    prefix_sum[i + 1] = prefix_sum[i] + t[i]

# Initialize DP
dp = [dict() for _ in range(n + 1)]
dp[0][0] = 0  # 0 orders processed, last shipment day 0, total dissatisfaction 0

for i in range(n + 1):
    if not dp[i]:
        continue
    for last_day, total_d in list(dp[i].items()):
        max_j = min(k, n - i)
        for j in range(1, max_j + 1):
            # Calculate max_T for the group
            max_T = t[i + j - 1]
            # Determine the earliest possible shipment day S
            if last_day == 0:
                S = max_T
            else:
                S = max(max_T, last_day + x)
            # Calculate the sum of T's in this group
            sum_T = prefix_sum[i + j] - prefix_sum[i]
            # Calculate added dissatisfaction
            added_d = j * S - sum_T
            # New state
            new_i = i + j
            new_d = total_d + added_d
            new_last_day = S
            # Update the DP table
            if new_last_day in dp[new_i]:
                if new_d < dp[new_i][new_last_day]:
                    dp[new_i][new_last_day] = new_d
            else:
                dp[new_i][new_last_day] = new_d

# The minimal total dissatisfaction is the minimum value in dp[n]
if not dp[n]:
    print(0)
else:
    print(min(dp[n].values()))