# YOUR CODE HERE
H, W, N = map(int, input().split())
holes = set(tuple(map(int, input().split())) for _ in range(N))

# Initialize the DP table
dp = [[0] * (W + 1) for _ in range(H + 1)]

# Fill the DP table
for i in range(1, H + 1):
    for j in range(1, W + 1):
        if (i, j) not in holes:
            dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

# Calculate the total number of holeless squares
result = sum(sum(row) for row in dp)

print(result)