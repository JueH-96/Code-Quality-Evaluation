from typing import List

class Solution:
	def maximumAmount(self, coins: List[List[int]]) -> int:
		m = len(coins)
		n = len(coins[0])
		NEG_INF = -10**18
		dp = [[[NEG_INF] * 3 for _ in range(n)] for __ in range(m)]
		
		if coins[0][0] >= 0:
			dp[0][0][0] = coins[0][0]
		else:
			dp[0][0][0] = coins[0][0]
			dp[0][0][1] = 0
		
		for i in range(m):
			for j in range(n):
				if i == 0 and j == 0:
					continue
				for k in range(3):
					if coins[i][j] >= 0:
						candidates = []
						if i > 0:
							candidates.append(dp[i-1][j][k])
						if j > 0:
							candidates.append(dp[i][j-1][k])
						if candidates:
							dp[i][j][k] = max(candidates) + coins[i][j]
					else:
						candidate1 = NEG_INF
						candidates1 = []
						if i > 0:
							candidates1.append(dp[i-1][j][k])
						if j > 0:
							candidates1.append(dp[i][j-1][k])
						if candidates1:
							candidate1 = max(candidates1) + coins[i][j]
						
						candidate2 = NEG_INF
						if k >= 1:
							candidates2 = []
							if i > 0:
								candidates2.append(dp[i-1][j][k-1])
							if j > 0:
								candidates2.append(dp[i][j-1][k-1])
							if candidates2:
								candidate2 = max(candidates2)
						
						dp[i][j][k] = max(candidate1, candidate2)
		
		return max(dp[m-1][n-1])