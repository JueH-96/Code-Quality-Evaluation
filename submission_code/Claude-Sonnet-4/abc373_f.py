N, W = map(int, input().split())
items = []
for _ in range(N):
    w, v = map(int, input().split())
    items.append((w, v))

# For each item type, find reasonable upper bound for number of items to consider
max_items = []
for w, v in items:
    # Happiness = k*v - k^2
    # This is maximized at k = v/2, but we need to consider integer constraints
    # and the fact that happiness becomes negative for large k
    # Let's consider up to the point where happiness becomes non-positive
    # k*v - k^2 <= 0 when k >= v
    # Also limit by weight constraint
    upper_bound = min(v, W // w + 1)
    max_items.append(upper_bound)

# DP: dp[i][w] = maximum happiness using first i item types with weight exactly w
# We'll use a 1D DP array and update it for each item type
dp = [-float('inf')] * (W + 1)
dp[0] = 0

for i in range(N):
    w_i, v_i = items[i]
    new_dp = dp[:]
    
    for weight in range(W + 1):
        if dp[weight] == -float('inf'):
            continue
            
        # Try taking k items of type i
        for k in range(1, max_items[i] + 1):
            new_weight = weight + k * w_i
            if new_weight <= W:
                happiness = k * v_i - k * k
                new_dp[new_weight] = max(new_dp[new_weight], dp[weight] + happiness)
    
    dp = new_dp

# Find maximum happiness
result = max(dp)
print(result)