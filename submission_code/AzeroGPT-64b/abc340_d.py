n = int(input())
a = [0] * n
b = [0] * (n-1)
x = [0] * (n-1)
for i in range(n-1):
    a[i], b[i], x[i] = list(map(int,input().split()))
a[-1] = 0

dp = [float("inf")] * n
dp[0] = 0

for i in range(n-1):
    dp[i+1] = min(dp[i+1], dp[i] + a[i])
    if dp[x[i] - 1] > dp[i] + b[i]:
        dp[x[i] - 1] = dp[i] + b[i]
        for j in range(x[i] - 1, n):
            dp[j] = min(dp[j], dp[j-1] + a[j-1])

print(dp[-1])