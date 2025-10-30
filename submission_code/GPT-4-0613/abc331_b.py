N, S, M, L = map(int, input().split())
dp = [10**9]*(N+1)
dp[0] = 0
for i in range(6, N+1):
    dp[i] = min(dp[i], dp[i-6]+S)
for i in range(8, N+1):
    dp[i] = min(dp[i], dp[i-8]+M)
for i in range(12, N+1):
    dp[i] = min(dp[i], dp[i-12]+L)
print(min(dp[N:]))