class Solution:
	def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
		horizontalCut.sort(reverse=True)
		verticalCut.sort(reverse=True)
		
		dp = [[0] * n for _ in range(m)]
		
		for i in range(m):
			for j in range(n):
				if i == 0 and j == 0:
					continue
				options = []
				if i > 0:
					options.append(dp[i-1][j] + horizontalCut[i-1] * (j + 1))
				if j > 0:
					options.append(dp[i][j-1] + verticalCut[j-1] * (i + 1))
				dp[i][j] = min(options)
		
		return dp[m-1][n-1]