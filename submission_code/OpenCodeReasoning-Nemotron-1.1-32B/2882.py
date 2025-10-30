mod = 10**9 + 7

class Solution:
	def numberOfWays(self, n: int, x: int) -> int:
		base = 1
		while (base + 1) ** x <= n:
			base += 1
		
		dp = [0] * (n + 1)
		dp[0] = 1
		
		for i in range(1, base + 1):
			power_val = i ** x
			for j in range(n, power_val - 1, -1):
				dp[j] = (dp[j] + dp[j - power_val]) % mod
		
		return dp[n] % mod