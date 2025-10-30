N, K, X = map(int, input().split())
T = list(map(int, input().split()))

dp = [float('inf')] * (N + 1)
last_day = [0] * (N + 1)
dp[0] = 0
last_day[0] = 0

for i in range(1, N + 1):
    for j in range(max(0, i - K), i):
        # Ship orders j+1 to i together
        if j == 0:
            ship_day = T[i-1]  # No previous shipment
        else:
            ship_day = max(T[i-1], last_day[j] + X)
        
        cost = 0
        for k in range(j, i):
            cost += ship_day - T[k]
        
        if dp[j] + cost < dp[i]:
            dp[i] = dp[j] + cost
            last_day[i] = ship_day

print(dp[N])