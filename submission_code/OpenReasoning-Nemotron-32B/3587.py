from typing import List

class Solution:
	def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
		NEG = -10**9
		dp = [NEG] * n
		
		for j in range(n):
			best = stayScore[0][j]
			for l in range(n):
				if l == j:
					continue
				if travelScore[l][j] > best:
					best = travelScore[l][j]
			dp[j] = best
		
		for i in range(1, k):
			dp_next = [NEG] * n
			for j in range(n):
				option1 = dp[j] + stayScore[i][j]
				option2 = NEG
				for l in range(n):
					if l == j:
						continue
					total = dp[l] + travelScore[l][j]
					if total > option2:
						option2 = total
				dp_next[j] = max(option1, option2)
			dp = dp_next
		
		return max(dp)