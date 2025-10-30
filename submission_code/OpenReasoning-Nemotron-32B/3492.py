from typing import List

class Solution:
	def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
		m = len(grid)
		n = len(grid[0])
		d = [[0] * (n + 1) for _ in range(m + 1)]
		non_dot = [[0] * (n + 1) for _ in range(m + 1)]
		count = 0
		
		for i in range(1, m + 1):
			for j in range(1, n + 1):
				char = grid[i - 1][j - 1]
				add_d = 0
				add_nd = 0
				if char == 'X':
					add_d = 1
					add_nd = 1
				elif char == 'Y':
					add_d = -1
					add_nd = 1
				
				d[i][j] = d[i - 1][j] + d[i][j - 1] - d[i - 1][j - 1] + add_d
				non_dot[i][j] = non_dot[i - 1][j] + non_dot[i][j - 1] - non_dot[i - 1][j - 1] + add_nd
				
				if d[i][j] == 0 and non_dot[i][j] >= 2:
					count += 1
		
		return count