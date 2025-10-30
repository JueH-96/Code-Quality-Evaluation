class Solution:
	def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
		mod = 10**9 + 7
		m, n = len(grid), len(grid[0])
		dp = [[[0] * 16 for _ in range(n)] for __ in range(m)]
		dp[0][0][grid[0][0]] = 1
		
		for i in range(m):
			for j in range(n):
				num = grid[i][j]
				if i == 0 and j == 0:
					continue
				for x in range(16):
					total = 0
					if i - 1 >= 0:
						total = (total + dp[i-1][j][x ^ num]) % mod
					if j - 1 >= 0:
						total = (total + dp[i][j-1][x ^ num]) % mod
					dp[i][j][x] = total
		
		return dp[m-1][n-1][k] % mod