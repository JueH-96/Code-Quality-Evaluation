class Solution:
	def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
		horizontalCut.sort()
		verticalCut.sort()
		total_fixed = sum(horizontalCut) + sum(verticalCut)
		dp = [[0] * n for _ in range(m)]
		
		for i in range(m):
			for j in range(n):
				if i == 0 and j == 0:
					dp[i][j] = 0
				else:
					option1 = float('inf')
					option2 = float('inf')
					if i > 0:
						option1 = dp[i-1][j] + horizontalCut[i-1] * j
					if j > 0:
						option2 = dp[i][j-1] + verticalCut[j-1] * i
					dp[i][j] = min(option1, option2)
		
		return total_fixed + dp[m-1][n-1]