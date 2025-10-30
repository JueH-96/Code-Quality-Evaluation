# YOUR CODE HERE
n, W = map(int, input().split())
items = []
for _ in range(n):
    w, v = map(int, input().split())
    items.append((w, v))

# dp[i][j] = max happiness using first i items with weight at most j
dp = [[0] * (W + 1) for _ in range(n + 1)]

for i in range(1, n + 1):
    w, v = items[i - 1]
    for j in range(W + 1):
        dp[i][j] = dp[i - 1][j]  # Don't take any of item i
        # Try all possible counts of item i
        max_k = min((v + 1) // 2, j // w)
        for k in range(1, max_k + 1):
            dp[i][j] = max(dp[i][j], dp[i - 1][j - k * w] + k * v - k * k)

print(dp[n][W])