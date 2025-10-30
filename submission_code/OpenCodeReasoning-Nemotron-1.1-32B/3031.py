class Solution:
	def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
		n = len(grid)
		m = len(grid[0])
		mod = 12345
		
		if n == 0 or m == 0:
			return grid
		
		row_products = [1] * n
		for i in range(n):
			prod = 1
			for j in range(m):
				prod = (prod * grid[i][j]) % mod
			row_products[i] = prod
		
		total_above = [1] * n
		for i in range(1, n):
			total_above[i] = (total_above[i-1] * row_products[i-1]) % mod
		
		total_below = [1] * n
		for i in range(n-2, -1, -1):
			total_below[i] = (total_below[i+1] * row_products[i+1]) % mod
		
		res = [[0] * m for _ in range(n)]
		
		for i in range(n):
			prefix = [1] * m
			for j in range(1, m):
				prefix[j] = (prefix[j-1] * grid[i][j-1]) % mod
			
			suffix = [1] * m
			for j in range(m-2, -1, -1):
				suffix[j] = (suffix[j+1] * grid[i][j+1]) % mod
			
			for j in range(m):
				without_j = (prefix[j] * suffix[j]) % mod
				res[i][j] = (total_above[i] * total_below[i] % mod * without_j) % mod
		
		return res