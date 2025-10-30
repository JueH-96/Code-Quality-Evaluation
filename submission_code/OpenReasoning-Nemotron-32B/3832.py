class Solution:
	def minCosts(self, cost: List[int]) -> List[int]:
		n = len(cost)
		ans = [0] * n
		ans[0] = cost[0]
		for i in range(1, n):
			ans[i] = min(ans[i-1], cost[i])
		return ans