s = input().strip()
n = len(s)
dp = [0] * (n + 1)
dp[0] = 0
for i in range(1, n + 1):
	dp[i] = dp[i - 1] + 1
	if i >= 2 and s[i - 1] == '0' and s[i - 2] == '0':
		dp[i] = min(dp[i], dp[i - 2] + 1)
print(dp[n])