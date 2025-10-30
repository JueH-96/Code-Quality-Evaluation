class Solution:
	def minCost(self, nums: List[int]) -> int:
		n = len(nums)
		dp = [0] + [float('inf')] * n
		for i in range(n):
			if i + 1 <= n:
				dp[i+1] = min(dp[i+1], dp[i] + nums[i])
			if i + 2 <= n:
				dp[i+2] = min(dp[i+2], dp[i] + max(nums[i], nums[i+1]))
			if i + 3 <= n:
				dp[i+3] = min(dp[i+3], dp[i] + max(nums[i], nums[i+1], nums[i+2]))
		return dp[n]