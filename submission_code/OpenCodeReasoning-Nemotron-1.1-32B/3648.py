class Solution:
	def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
		n = len(fruits)
		NEG_INF = -10**18
		dp_prev = [[NEG_INF] * n for _ in range(n)]
		dp_prev[n-1][n-1] = fruits[0][0] + fruits[0][n-1] + fruits[n-1][0]
		
		for t in range(1, n):
			dp_curr = [[NEG_INF] * n for _ in range(n)]
			low_bound = max(0, n-1 - t)
			high_bound = min(n-1, n-1 + t)
			for j2 in range(low_bound, high_bound + 1):
				for i3 in range(low_bound, high_bound + 1):
					for dj2 in (-1, 0, 1):
						for di3 in (-1, 0, 1):
							j2_prev = j2 - dj2
							i3_prev = i3 - di3
							if not (0 <= j2_prev < n and 0 <= i3_prev < n):
								continue
							if dp_prev[j2_prev][i3_prev] == NEG_INF:
								continue
							add = fruits[t][t]
							if j2 > t:
								add += fruits[t][j2]
							if i3 > t:
								add += fruits[i3][t]
							new_val = dp_prev[j2_prev][i3_prev] + add
							if new_val > dp_curr[j2][i3]:
								dp_curr[j2][i3] = new_val
			dp_prev = dp_curr
		
		return dp_prev[n-1][n-1]