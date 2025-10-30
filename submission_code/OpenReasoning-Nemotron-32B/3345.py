class Solution:
	def sumOfPower(self, nums: List[int], k: int) -> int:
		mod = 10**9 + 7
		dp = [0] * (k + 1)
		dp[0] = 1
		for num in nums:
			for s in range(k, -1, -1):
				if s >= num:
					dp[s] = (2 * dp[s] + dp[s - num]) % mod
				else:
					dp[s] = (2 * dp[s]) % mod
		return dp[k] % mod