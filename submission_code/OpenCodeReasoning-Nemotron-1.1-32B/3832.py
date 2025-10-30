class Solution:
	def minCosts(self, cost: List[int]) -> List[int]:
		min_val = float('inf')
		dp = []
		for c in cost:
			current = min(c, min_val)
			dp.append(current)
			min_val = min(min_val, current)
		return dp