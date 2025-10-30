s = input().strip()
n = len(s)
if n == 0:
	print(0)
else:
	dp = [0] * (n + 1)
	for i in range(1, n + 1):
		dp[i] = dp[i - 1] + 1
		if i >= 2 and s[i - 2:i] == "00":
			dp[i] = min(dp[i], dp[i - 2] + 1)
	print(dp[n])