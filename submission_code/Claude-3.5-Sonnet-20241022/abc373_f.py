N, W = map(int, input().split())
items = []
for _ in range(N):
    w, v = map(int, input().split())
    items.append((w, v))

# dp[i][w] = max happiness using first i items with total weight w
dp = [[-float('inf')] * (W + 1) for _ in range(N + 1)]
dp[0][0] = 0

for i in range(N):
    w, v = items[i]
    for curr_w in range(W + 1):
        # Don't take any of item i
        dp[i + 1][curr_w] = dp[i][curr_w]
        
        # Try taking k items of type i
        k = 1
        while k * w + curr_w <= W:
            happiness = k * v - k * k  # happiness formula
            if dp[i][curr_w] != -float('inf'):
                dp[i + 1][k * w + curr_w] = max(
                    dp[i + 1][k * w + curr_w],
                    dp[i][curr_w] + happiness
                )
            k += 1

print(max(dp[N]))