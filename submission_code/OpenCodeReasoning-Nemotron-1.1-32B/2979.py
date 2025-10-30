from collections import defaultdict
from typing import List

class Solution:
	def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
		offers_by_end = defaultdict(list)
		for s, e, g in offers:
			offers_by_end[e].append((s, g))
		
		dp = [0] * n
		
		for i in range(n):
			if i > 0:
				dp[i] = dp[i-1]
			if i in offers_by_end:
				for s, g in offers_by_end[i]:
					prev = dp[s-1] if s > 0 else 0
					total = prev + g
					if total > dp[i]:
						dp[i] = total
		return dp[n-1]