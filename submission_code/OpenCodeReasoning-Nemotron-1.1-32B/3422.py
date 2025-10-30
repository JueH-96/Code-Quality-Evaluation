class Solution:
	def valueAfterKSeconds(self, n: int, k: int) -> int:
		mod = 10**9 + 7
		dp = [1] * n
		for _ in range(k):
			for i in range(1, n):
				dp[i] = (dp[i-1] + dp[i]) % mod
		return dp[-1] % mod