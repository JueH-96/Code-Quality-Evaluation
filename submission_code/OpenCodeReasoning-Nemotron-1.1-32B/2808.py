from typing import List

class Solution:
	def paintWalls(self, cost: List[int], time: List[int]) -> int:
		n = len(cost)
		INF = 10**18
		dp = [[INF] * 501 for _ in range(n+1)]
		dp[0][0] = 0
		
		for i in range(n):
			for count in range(n-1, -1, -1):
				for t in range(500, -1, -1):
					if dp[count][t] != INF:
						new_count = count + 1
						new_t = t + time[i]
						if new_t > 500:
							new_t = 500
						new_cost = dp[count][t] + cost[i]
						if new_cost < dp[new_count][new_t]:
							dp[new_count][new_t] = new_cost
		
		ans = INF
		for count in range(n+1):
			for t in range(501):
				if dp[count][t] < INF and t >= n - count:
					ans = min(ans, dp[count][t])
		return ans