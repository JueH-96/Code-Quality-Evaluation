# YOUR CODE HERE
n, k, x = map(int, input().split())
t = list(map(int, input().split()))

# dp[i] = minimum dissatisfaction for shipping orders 0 to i-1
dp = [float('inf')] * (n + 1)
dp[0] = 0

# last_ship[i] = day of last shipment when we've shipped orders 0 to i-1
last_ship = [-float('inf')] * (n + 1)

for i in range(1, n + 1):
    # Try shipping orders j to i-1 together
    for j in range(max(0, i - k), i):
        # Can only ship at most k orders together
        if i - j > k:
            continue
            
        # Earliest we can ship these orders
        earliest_order_day = t[i-1]  # Latest order in the batch
        
        # Earliest we can ship based on previous shipment
        earliest_ship_day = last_ship[j] + x
        
        # Actual shipping day
        ship_day = max(earliest_order_day, earliest_ship_day)
        
        # Calculate dissatisfaction for orders j to i-1
        dissatisfaction = 0
        for idx in range(j, i):
            dissatisfaction += ship_day - t[idx]
        
        # Update dp if this is better
        if dp[j] + dissatisfaction < dp[i]:
            dp[i] = dp[j] + dissatisfaction
            last_ship[i] = ship_day

print(dp[n])