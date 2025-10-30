# YOUR CODE HERE
N, K, X = map(int, input().split())
T = list(map(int, input().split()))

# dp[i] = minimum dissatisfaction to ship first i orders
# We also need to track the last shipment day
# dp[i] = list of (dissatisfaction, last_shipment_day) for shipping first i orders

dp = [[] for _ in range(N + 1)]
dp[0] = [(0, 0)]  # (dissatisfaction, last_shipment_day)

for i in range(1, N + 1):
    dp[i] = []
    candidates = {}
    
    # Try all possible ways to ship orders ending at position i
    for j in range(1, min(K, i) + 1):
        # Ship orders from i-j to i-1 (0-indexed)
        start_idx = i - j
        
        # For each possible state of shipping first start_idx orders
        for prev_dissatisfaction, last_ship_day in dp[start_idx]:
            # Latest order day in current shipment
            latest_order_day = T[i - 1]
            
            # Earliest possible shipment day
            if start_idx == 0:
                ship_day = latest_order_day
            else:
                ship_day = max(latest_order_day, last_ship_day + X)
            
            # Calculate dissatisfaction for current shipment
            current_dissatisfaction = 0
            for k in range(start_idx, i):
                current_dissatisfaction += ship_day - T[k]
            
            total_dissatisfaction = prev_dissatisfaction + current_dissatisfaction
            
            # Keep only the best result for each last_ship_day
            if ship_day not in candidates or total_dissatisfaction < candidates[ship_day]:
                candidates[ship_day] = total_dissatisfaction
    
    # Convert candidates to list and keep only non-dominated solutions
    for ship_day, dissatisfaction in candidates.items():
        dp[i].append((dissatisfaction, ship_day))
    
    # Remove dominated solutions (higher dissatisfaction and higher/equal ship day)
    dp[i].sort()
    filtered = []
    for dissatisfaction, ship_day in dp[i]:
        if not filtered or ship_day < filtered[-1][1]:
            filtered.append((dissatisfaction, ship_day))
    dp[i] = filtered

# Find minimum dissatisfaction for shipping all orders
result = min(dissatisfaction for dissatisfaction, _ in dp[N])
print(result)