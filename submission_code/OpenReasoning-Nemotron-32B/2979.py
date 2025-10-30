from typing import List

class Solution:
	def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
		offers_by_end = [[] for _ in range(n)]
		for s, e, g in offers:
			offers_by_end[e].append((s, g))
		
		dp = [0] * (n + 1)
		for i in range(1, n + 1):
			dp[i] = dp[i - 1]
			for s, g in offers_by_end[i - 1]:
				candidate = dp[s] + g
				if candidate > dp[i]:
					dp[i] = candidate
		return dp[n]