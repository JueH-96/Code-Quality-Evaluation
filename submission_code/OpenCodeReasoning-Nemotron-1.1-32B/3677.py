class Solution:
	def maximumAmount(self, coins: List[List[int]]) -> int:
		NEG_INF = -10**18
		m = len(coins)
		n = len(coins[0])
		dp = [[[NEG_INF] * 3 for _ in range(n)] for __ in range(m)]
		
		dp[0][0][0] = coins[0][0]
		if coins[0][0] < 0:
			dp[0][0][1] = 0
		
		for i in range(m):
			for j in range(n):
				if i == 0 and j == 0:
					continue
				for k in range(3):
					best_prev = NEG_INF
					if i > 0:
						best_prev = max(best_prev, dp[i-1][j][k])
					if j > 0:
						best_prev = max(best_prev, dp[i][j-1][k])
					
					if coins[i][j] >= 0:
						current_val = best_prev + coins[i][j]
						if current_val > dp[i][j][k]:
							dp[i][j][k] = current_val
					else:
						option1 = best_prev + coins[i][j]
						option2 = NEG_INF
						if k >= 1:
							best_prev_k1 = NEG_INF
							if i > 0:
								best_prev_k1 = max(best_prev_k1, dp[i-1][j][k-1])
							if j > 0:
								best_prev_k1 = max(best_prev_k1, dp[i][j-1][k-1])
							option2 = best_prev_k1
						current_val = max(option1, option2)
						if current_val > dp[i][j][k]:
							dp[i][j][k] = current_val
		
		return max(dp[m-1][n-1])