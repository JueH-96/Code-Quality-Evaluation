class Solution:
	def maxSum(self, nums: List[int], k: int, m: int) -> int:
		n = len(nums)
		NEG_INF = -10**18
		P = [0] * (n + 1)
		for i in range(n):
			P[i+1] = P[i] + nums[i]
		
		dp = [[NEG_INF] * (k + 1) for _ in range(n + 1)]
		for i in range(n + 1):
			dp[i][0] = 0
			
		best = [NEG_INF] * (k + 1)
		
		for i in range(n):
			if i >= m - 1:
				s0 = i - m + 1
				for j in range(1, k + 1):
					val = dp[s0][j-1] - P[s0]
					if val > best[j]:
						best[j] = val
			for j in range(1, k + 1):
				total = best[j] + P[i+1]
				dp[i+1][j] = max(dp[i][j], total)
				
		return dp[n][k]