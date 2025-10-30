from collections import defaultdict

class Solution:
	def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
		n = len(grid)
		diag_vals = defaultdict(list)
		
		for i in range(n):
			for j in range(n):
				d = i - j
				diag_vals[d].append(grid[i][j])
				
		for d in diag_vals:
			if d >= 0:
				diag_vals[d].sort(reverse=True)
			else:
				diag_vals[d].sort()
				
		result = [[0] * n for _ in range(n)]
		
		for d, vals in diag_vals.items():
			i_min = max(0, d)
			i_max = min(n-1, n-1 + d)
			idx = 0
			for i in range(i_min, i_max + 1):
				j = i - d
				result[i][j] = vals[idx]
				idx += 1
				
		return result