MOD = 10**9 + 7

class Solution:
	def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
		n = len(nums)
		T = max(nums)
		total = sum(nums)
		total_deficit = n * T - total
		if total_deficit == 0:
			return 0
		if cost2 >= 2 * cost1:
			return (cost1 * total_deficit) % MOD
		else:
			min_val = min(nums)
			max_d = T - min_val
			term1 = total_deficit // 2
			term2 = total_deficit - max_d
			S_max = 2 * min(term1, term2)
			cost_val = cost1 * (total_deficit - S_max) + (cost2 * S_max) // 2
			return cost_val % MOD