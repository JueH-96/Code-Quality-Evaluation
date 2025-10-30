n = int(input())
a = list(map(int, input().split()))

# dp[i][j] = max experience from first i monsters, defeating exactly j
dp = [[-1] * (n + 1) for _ in range(n + 1)]
dp[0][0] = 0

for i in range(1, n + 1):
    for j in range(i + 1):
        # Skip monster i
        if dp[i-1][j] >= 0:
            dp[i][j] = max(dp[i][j], dp[i-1][j])
        
        # Defeat monster i (if j > 0)
        if j > 0 and dp[i-1][j-1] >= 0:
            exp_gain = a[i-1] * (2 if j % 2 == 0 else 1)
            dp[i][j] = max(dp[i][j], dp[i-1][j-1] + exp_gain)

print(max(dp[n][j] for j in range(n + 1) if dp[n][j] >= 0))