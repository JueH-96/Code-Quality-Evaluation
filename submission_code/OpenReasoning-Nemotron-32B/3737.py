from typing import List

class Solution:
	def minCost(self, n: int, cost: List[List[int]]) -> int:
		k = n // 2
		INF = 10**18
		dp = [[INF] * 3 for _ in range(3)]
		
		for c0 in range(3):
			for c1 in range(3):
				if c0 != c1:
					dp[c0][c1] = cost[0][c0] + cost[n-1][c1]
		
		for i in range(1, k):
			new_dp = [[INF] * 3 for _ in range(3)]
			for a in range(3):
				for b in range(3):
					if dp[a][b] == INF:
						continue
					for c in range(3):
						if c == a:
							continue
						for d in range(3):
							if d == b or c == d:
								continue
							total_cost = dp[a][b] + cost[i][c] + cost[n-1-i][d]
							if total_cost < new_dp[c][d]:
								new_dp[c][d] = total_cost
			dp = new_dp
		
		ans = INF
		for a in range(3):
			for b in range(3):
				if dp[a][b] < ans:
					ans = dp[a][b]
		return ans