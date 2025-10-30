from typing import List

class Solution:
	def maxScore(self, grid: List[List[int]]) -> int:
		n = len(grid)
		dp = {frozenset(): 0}
		for i in range(n):
			row_vals = set(grid[i])
			new_dp = dp.copy()
			for s, total in dp.items():
				for v in row_vals:
					if v not in s:
						new_set = s | {v}
						new_total = total + v
						if new_set not in new_dp:
							new_dp[new_set] = new_total
			dp = new_dp
		ans = 0
		for s, total in dp.items():
			if s:
				ans = max(ans, total)
		return ans