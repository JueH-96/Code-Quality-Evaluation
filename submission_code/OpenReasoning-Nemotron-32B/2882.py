class Solution:
	def numberOfWays(self, n: int, x: int) -> int:
		mod = 10**9 + 7
		dp = [0] * (n + 1)
		dp[0] = 1
		candidates = []
		i = 1
		while True:
			power_val = i ** x
			if power_val > n:
				break
			candidates.append(power_val)
			i += 1
		
		for c in candidates:
			for j in range(n, c - 1, -1):
				dp[j] = (dp[j] + dp[j - c]) % mod
		
		return dp[n] % mod