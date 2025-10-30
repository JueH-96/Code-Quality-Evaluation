# Read input
H, W, N = map(int, input().split())

holes = set()
for _ in range(N):
    a, b = map(int, input().split())
    holes.add((a, b))

# Initialize DP table
dp = [[0] * W for _ in range(H)]

# Process from bottom-right to top-left
for i in reversed(range(H)):
    for j in reversed(range(W)):
        if (i + 1, j + 1) in holes:
            dp[i][j] = 0
        else:
            right = dp[i][j + 1] if j + 1 < W else 0
            down = dp[i + 1][j] if i + 1 < H else 0
            diag = dp[i + 1][j + 1] if (i + 1 < H) and (j + 1 < W) else 0
            dp[i][j] = 1 + min(right, down, diag)

# Calculate the total number of holeless squares
total = sum(sum(row) for row in dp)
print(total)