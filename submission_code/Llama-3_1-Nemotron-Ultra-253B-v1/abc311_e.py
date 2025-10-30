H, W, N = map(int, input().split())
holed = set()
for _ in range(N):
    a, b = map(int, input().split())
    holed.add((a, b))

# Initialize DP table with 0s, and with (H+2) rows and (W+2) columns
dp = [[0] * (W + 2) for _ in range(H + 2)]

for i in range(H, 0, -1):
    for j in range(W, 0, -1):
        if (i, j) in holed:
            dp[i][j] = 0
        else:
            dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])

total = 0
for i in range(1, H + 1):
    for j in range(1, W + 1):
        total += dp[i][j]

print(total)