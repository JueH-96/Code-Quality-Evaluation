class Solution:
	def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
		rows = len(grid)
		cols = len(grid[0])
		
		row_ones = [sum(row) for row in grid]
		
		col_ones = [0] * cols
		for j in range(cols):
			for i in range(rows):
				col_ones[j] += grid[i][j]
		
		total_triangles = 0
		for i in range(rows):
			for j in range(cols):
				if grid[i][j] == 1:
					total_triangles += (row_ones[i] - 1) * (col_ones[j] - 1)
		
		return total_triangles