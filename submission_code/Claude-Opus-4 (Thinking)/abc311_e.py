# YOUR CODE HERE
H, W, N = map(int, input().split())
holed = set()
for _ in range(N):
    a, b = map(int, input().split())
    holed.add((a, b))

# dp[i][j] = maximum size of holeless square with top-left at (i, j)
# Using extra row/column for boundary conditions
dp = [[0] * (W + 2) for _ in range(H + 2)]

# Fill from bottom-right to top-left
for i in range(H, 0, -1):
    for j in range(W, 0, -1):
        if (i, j) in holed:
            dp[i][j] = 0
        else:
            dp[i][j] = min(dp[i+1][j], dp[i][j+1], dp[i+1][j+1]) + 1

# Count total holeless squares
total = 0
for i in range(1, H + 1):
    for j in range(1, W + 1):
        total += dp[i][j]

print(total)