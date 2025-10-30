H, W, N = map(int, input().split())

# Initialize grid with True (no holes)
grid = [[True for _ in range(W)] for _ in range(H)]

# Mark holes
for _ in range(N):
    a, b = map(int, input().split())
    grid[a-1][b-1] = False

dp = [[0 for _ in range(W)] for _ in range(H)]
total = 0

for i in range(H):
    for j in range(W):
        if grid[i][j]:
            if i == 0 or j == 0:
                dp[i][j] = 1
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        else:
            dp[i][j] = 0
        total += dp[i][j]

print(total)