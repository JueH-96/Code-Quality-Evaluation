class Solution:
	def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
		mod = 10**9 + 7
		m = len(grid)
		n = len(grid[0])
		
		if m == 1 and n == 1:
			return 1 if grid[0][0] == k else 0
		
		dp = [[0] * 16 for _ in range(n)]
		dp[0][grid[0][0]] = 1
		
		for j in range(1, n):
			for x in range(16):
				prev = x ^ grid[0][j]
				dp[j][x] = dp[j-1][prev] % mod
		
		for i in range(1, m):
			new_dp = [[0] * 16 for _ in range(n)]
			for x in range(16):
				prev = x ^ grid[i][0]
				new_dp[0][x] = dp[0][prev] % mod
			
			for j in range(1, n):
				for x in range(16):
					total = 0
					prev1 = x ^ grid[i][j]
					total = (total + dp[j][prev1]) % mod
					prev2 = x ^ grid[i][j]
					total = (total + new_dp[j-1][prev2]) % mod
					new_dp[j][x] = total
			dp = new_dp
		
		return dp[n-1][k] % mod