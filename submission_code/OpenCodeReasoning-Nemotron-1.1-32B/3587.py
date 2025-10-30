class Solution:
	def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
		prev = stayScore[0][:]
		for i in range(1, k):
			curr = [0] * n
			for j in range(n):
				stay_opt = prev[j] + stayScore[i][j]
				move_opt = -10**18
				for l in range(n):
					if l == j:
						continue
					candidate = prev[l] + travelScore[l][j]
					if candidate > move_opt:
						move_opt = candidate
				curr[j] = max(stay_opt, move_opt)
			prev = curr
		return max(prev)