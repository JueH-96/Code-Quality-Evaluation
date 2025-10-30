N, D, P = map(int, input().split())
F = list(map(int, input().split()))

dp = [0] * (N + 1)
F.append(float('inf'))

for i in range(N):
    dp[i + 1] = min(dp[i] + F[i], dp[i - D + 1] + P) if i >= D - 1 else min(dp[i] + F[i], P)

print(dp[N])