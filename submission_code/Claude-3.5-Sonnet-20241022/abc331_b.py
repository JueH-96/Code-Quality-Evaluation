N, S, M, L = map(int, input().split())

INF = float('inf')
dp = [INF] * (N + 12)  # Add buffer for the largest pack size
dp[0] = 0

for i in range(N + 12):
    if i >= 6:
        dp[i] = min(dp[i], dp[i-6] + S)
    if i >= 8:
        dp[i] = min(dp[i], dp[i-8] + M)
    if i >= 12:
        dp[i] = min(dp[i], dp[i-12] + L)

print(min(dp[N:N+12]))