N, S, M, L = map(int, input().split())
dp = [float('inf')] * (N+1)

dp[0] = 0
for i in range(N+1):
    if i >= 6:
        dp[i] = min(dp[i], dp[i-6] + S)
    if i >= 8:
        dp[i] = min(dp[i], dp[i-8] + M)
    if i >= 12:
        dp[i] = min(dp[i], dp[i-12] + L)

print(dp[N])