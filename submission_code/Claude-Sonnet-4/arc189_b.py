n = int(input())
x = list(map(int, input().split()))

# Use DP where dp[i] is minimum sum for first i pieces
dp = [float('inf')] * (n + 1)
dp[0] = 0

for i in range(n):
    # Option 1: Take piece i as is
    dp[i + 1] = min(dp[i + 1], dp[i] + x[i])
    
    # Option 2: If we have at least 4 pieces starting from some previous position,
    # we can apply operation
    if i >= 3:
        # Try applying operation on pieces (i-3), (i-2), (i-1), i
        a, b, c, d = x[i-3], x[i-2], x[i-1], x[i]
        M = (a + d) / 2
        new_sum = a + (2*M - c) + (2*M - b) + d
        dp[i + 1] = min(dp[i + 1], dp[i - 3] + new_sum)

print(int(dp[n]))