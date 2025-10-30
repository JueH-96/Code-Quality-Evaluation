N, K, X = map(int, input().split())
T = list(map(int, input().split()))

# Compute prefix sums for efficiency
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i+1] = prefix_sum[i] + T[i]

# dp[i] = (minimum dissatisfaction, last shipment day) for shipping the first i orders
dp = [(0, -float('inf'))] + [(float('inf'), 0) for _ in range(N)]

for i in range(1, N+1):
    for j in range(1, min(K, i) + 1):
        prev_dissatisfaction, prev_ship_day = dp[i-j]
        ship_day = max(T[i-1], prev_ship_day + X)
        dissatisfaction = ship_day * j - (prefix_sum[i] - prefix_sum[i-j])
        total_dissatisfaction = prev_dissatisfaction + dissatisfaction
        if total_dissatisfaction < dp[i][0]:
            dp[i] = (total_dissatisfaction, ship_day)

print(dp[N][0])