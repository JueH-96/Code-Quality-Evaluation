N, A, X, Y = map(int, input().split())
dp = [0] * (A + 1)
for i in range(A, 0, -1):
    dp[i] = min(X + dp[(i + A - 1) // A], Y * (1 + (i - 1) / 6) + dp[i // 2])
print(dp[min(N, A)])