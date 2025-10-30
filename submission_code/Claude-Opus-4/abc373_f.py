# YOUR CODE HERE
N, W = map(int, input().split())
items = []
for _ in range(N):
    w, v = map(int, input().split())
    items.append((w, v))

# dp[i][j] = max happiness using first i types with weight exactly j
dp = [[-1] * (W + 1) for _ in range(N + 1)]
dp[0][0] = 0

for i in range(N):
    w_i, v_i = items[i]
    
    # Copy previous values
    for j in range(W + 1):
        if dp[i][j] != -1:
            dp[i + 1][j] = max(dp[i + 1][j], dp[i][j])
    
    # Try taking k items of type i
    # Since happiness = k*v_i - k^2, it's maximized around k = v_i/2
    # We only need to check reasonable values of k
    max_k = min(W // w_i, v_i)  # No point taking more than v_i items
    
    for j in range(W + 1):
        if dp[i][j] == -1:
            continue
            
        for k in range(1, max_k + 1):
            new_weight = j + k * w_i
            if new_weight <= W:
                happiness = k * v_i - k * k
                dp[i + 1][new_weight] = max(dp[i + 1][new_weight], dp[i][j] + happiness)

# Find maximum happiness
ans = 0
for j in range(W + 1):
    if dp[N][j] != -1:
        ans = max(ans, dp[N][j])

print(ans)