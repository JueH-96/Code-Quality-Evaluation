n, k = map(int, input().split())
a = list(map(int, input().split()))

dp = [0] * (k + 1)
dp[0] = 0
dp[1] = 0

for i in range(2, k + 1):
    if i % 2 == 0:
        dp[i] = dp[i-2] + (a[i-1] - a[i-2])
    else:
        dp[i] = min(dp[i-1], dp[i-2] + (a[i-1] - a[i-2]))

print(dp[k])