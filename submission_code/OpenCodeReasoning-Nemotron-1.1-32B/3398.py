class Solution:
	def canMakeSquare(self, grid: List[List[str]]) -> bool:
		squares = [
			[(0,0), (0,1), (1,0), (1,1)],
			[(0,1), (0,2), (1,1), (1,2)],
			[(1,0), (1,1), (2,0), (2,1)],
			[(1,1), (1,2), (2,1), (2,2)]
		]
		for sq in squares:
			count_b = 0
			for (i, j) in sq:
				if grid[i][j] == 'B':
					count_b += 1
			count_w = 4 - count_b
			if count_b >= 3 or count_w >= 3:
				return True
		return False