n = int(input())
a = list(map(int, input().split()))

dp = [[0, 0] for _ in range(n+1)]

for i in range(n):
    dp[i+1][0] = dp[i][0]
    dp[i+1][1] = dp[i][0] + a[i]
    if i % 2 == 0:
        dp[i+1][1] += a[i]
    if dp[i][1] + a[i] > dp[i+1][1]:
        dp[i+1][1] = dp[i][1] + a[i]
        if i % 2 == 0:
            dp[i+1][1] += a[i]

print(max(dp[n]))