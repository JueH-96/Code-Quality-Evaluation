class Solution:
	def minFlips(self, grid: List[List[int]]) -> int:
		m = len(grid)
		n = len(grid[0])
		
		total_row = 0
		for i in range(m):
			for j in range(n // 2):
				if grid[i][j] != grid[i][n - 1 - j]:
					total_row += 1
		
		total_col = 0
		for j in range(n):
			for i in range(m // 2):
				if grid[i][j] != grid[m - 1 - i][j]:
					total_col += 1
		
		return min(total_row, total_col)