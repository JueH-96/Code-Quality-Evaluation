class Solution:
	def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
		MOD = 10**9 + 7
		T = max(nums)
		D = 0
		M = 0
		for x in nums:
			d = T - x
			D += d
			if d > M:
				M = d
				
		if cost2 < 2 * cost1:
			k = min(D // 2, D - M)
			total_cost = D * cost1 - k * (2 * cost1 - cost2)
		else:
			total_cost = D * cost1
			
		return total_cost % MOD