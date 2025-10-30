from typing import List

class Solution:
	def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
		m = len(grid)
		n = len(grid[0])
		A_prev = [0] * n
		B_prev = [0] * n
		count = 0
		
		for i in range(m):
			A_curr = [0] * n
			B_curr = [0] * n
			for j in range(n):
				topA = A_prev[j]
				leftA = A_curr[j-1] if j-1 >= 0 else 0
				diagA = A_prev[j-1] if (i-1 >= 0 and j-1 >= 0) else 0
				A_curr[j] = (1 if grid[i][j] == 'X' else 0) + topA + leftA - diagA
				
				topB = B_prev[j]
				leftB = B_curr[j-1] if j-1 >= 0 else 0
				diagB = B_prev[j-1] if (i-1 >= 0 and j-1 >= 0) else 0
				B_curr[j] = (1 if grid[i][j] == 'Y' else 0) + topB + leftB - diagB
				
				if A_curr[j] == B_curr[j] and A_curr[j] >= 1:
					count += 1
					
			A_prev = A_curr
			B_prev = B_curr
			
		return count