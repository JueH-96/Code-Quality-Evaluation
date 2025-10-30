class Solution:
	def canMakeSquare(self, grid: List[List[str]]) -> bool:
		# Check the original grid for any uniform 2x2 square
		for r0 in range(2):
			for c0 in range(2):
				if grid[r0][c0] == grid[r0][c0+1] == grid[r0+1][c0] == grid[r0+1][c0+1]:
					return True
		
		# Try flipping each cell one by one
		for i in range(3):
			for j in range(3):
				flipped = 'W' if grid[i][j] == 'B' else 'B'
				squares_to_check = []
				if i < 2 and j < 2:
					squares_to_check.append((0, 0))
				if i < 2 and j >= 1:
					squares_to_check.append((0, 1))
				if i >= 1 and j < 2:
					squares_to_check.append((1, 0))
				if i >= 1 and j >= 1:
					squares_to_check.append((1, 1))
				
				for (r0, c0) in squares_to_check:
					a = grid[r0][c0] if (r0, c0) != (i, j) else flipped
					b = grid[r0][c0+1] if (r0, c0+1) != (i, j) else flipped
					c_val = grid[r0+1][c0] if (r0+1, c0) != (i, j) else flipped
					d = grid[r0+1][c0+1] if (r0+1, c0+1) != (i, j) else flipped
					if a == b == c_val == d:
						return True
		
		return False