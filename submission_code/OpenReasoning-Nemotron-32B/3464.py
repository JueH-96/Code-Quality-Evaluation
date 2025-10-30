from typing import List

class Solution:
	def maximumTotalCost(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		P = [0] * n
		P[0] = nums[0]
		for i in range(1, n):
			if i % 2 == 0:
				P[i] = P[i-1] + nums[i]
			else:
				P[i] = P[i-1] - nums[i]
		
		dp = [0] * n
		best_even = -10**18
		best_odd = -10**18
		
		for i in range(n):
			candidate = P[i]
			if i >= 1:
				candidate = max(candidate, best_even + P[i], best_odd - P[i])
			dp[i] = candidate
			
			if i < n-1:
				j0 = i + 1
				sign = 1 if j0 % 2 == 0 else -1
				A_j0 = dp[i] - sign * P[i]
				if j0 % 2 == 0:
					best_even = max(best_even, A_j0)
				else:
					best_odd = max(best_odd, A_j0)
		
		return dp[n-1]