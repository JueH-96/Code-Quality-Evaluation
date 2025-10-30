# Read the input
W, H, L, R, D, U = map(int, input().split())

# Calculate the number of possible paths
mod = 998244353
dp = [[0] * (W + 1) for _ in range(H + 1)]

# Initialize the starting points
for x in range(L, R + 1):
    for y in range(D, U + 1):
        dp[y][x] = 1

# Compute the number of paths
for y in range(H + 1):
    for x in range(W + 1):
        if x > 0:
            dp[y][x] = (dp[y][x] + dp[y][x - 1]) % mod
        if y > 0:
            dp[y][x] = (dp[y][x] + dp[y - 1][x]) % mod

# Print the answer
print(sum(sum(row) for row in dp) % mod)