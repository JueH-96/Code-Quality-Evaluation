class Solution:
	def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
		n = len(nums)
		prefix_sum = [0] * n
		prefix_cost = [0] * n
		prefix_sum[0] = nums[0]
		prefix_cost[0] = cost[0]
		for i in range(1, n):
			prefix_sum[i] = prefix_sum[i-1] + nums[i]
			prefix_cost[i] = prefix_cost[i-1] + cost[i]
		
		dp = [[10**18] * (n+1) for _ in range(n)]
		
		for i in range(n):
			dp[i][1] = (prefix_sum[i] + k * 1) * prefix_cost[i]
		
		if n == 1:
			return dp[0][1]
		
		for s_val in range(2, n+1):
			for i in range(s_val-1, n):
				A = prefix_sum[i] + k * s_val
				best = 10**18
				for j in range(s_val-1, i+1):
					seg_cost = prefix_cost[i] - prefix_cost[j-1]
					candidate = dp[j-1][s_val-1] + A * seg_cost
					if candidate < best:
						best = candidate
				dp[i][s_val] = best
		
		ans = min(dp[n-1][s_val] for s_val in range(1, n+1))
		return ans