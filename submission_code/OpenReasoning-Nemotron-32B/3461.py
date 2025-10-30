class Solution:
	def minimumArea(self, grid: List[List[int]]) -> int:
		rows = len(grid)
		cols = len(grid[0])
		min_row = rows
		max_row = -1
		min_col = cols
		max_col = -1
		
		for i in range(rows):
			row_min = cols
			row_max = -1
			has_one = False
			for j in range(cols):
				if grid[i][j] == 1:
					has_one = True
					if j < row_min:
						row_min = j
					if j > row_max:
						row_max = j
			if has_one:
				if i < min_row:
					min_row = i
				if i > max_row:
					max_row = i
				if row_min < min_col:
					min_col = row_min
				if row_max > max_col:
					max_col = row_max
		
		return (max_row - min_row + 1) * (max_col - min_col + 1)