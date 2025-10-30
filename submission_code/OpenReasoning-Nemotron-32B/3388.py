class Solution:
	def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
		m = len(grid)
		n = len(grid[0])
		row_ones = [sum(row) for row in grid]
		col_ones = [sum(col) for col in zip(*grid)]
		
		ans = 0
		for i in range(m):
			for j in range(n):
				if grid[i][j] == 1:
					ans += (row_ones[i] - 1) * (col_ones[j] - 1)
		return ans