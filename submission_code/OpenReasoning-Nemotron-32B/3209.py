class Solution:
	def minimumCoins(self, prices: List[int]) -> int:
		n = len(prices)
		INF = 10**15
		dp = [INF] * (n + 1)
		g = [-1] * (n + 1)
		
		dp[0] = 0
		g[0] = -1
		
		for i in range(n):
			if dp[i] == INF:
				continue
				
			for j in range(0, i):
				new_cov = min(2 * j + 1, n - 1)
				total_cov = max(g[i], new_cov)
				new_state = total_cov + 1
				if new_state > n:
					new_state = n
				new_cost = dp[i] + prices[j]
				if new_cost < dp[new_state]:
					dp[new_state] = new_cost
					g[new_state] = total_cov
				elif new_cost == dp[new_state] and total_cov > g[new_state]:
					g[new_state] = total_cov
			
			new_cov = min(2 * i + 1, n - 1)
			total_cov = max(g[i], new_cov)
			new_state = total_cov + 1
			if new_state > n:
				new_state = n
			new_cost = dp[i] + prices[i]
			if new_cost < dp[new_state]:
				dp[new_state] = new_cost
				g[new_state] = total_cov
			elif new_cost == dp[new_state] and total_cov > g[new_state]:
				g[new_state] = total_cov
				
			if g[i] >= i:
				if dp[i] < dp[i + 1]:
					dp[i + 1] = dp[i]
					g[i + 1] = g[i]
				elif dp[i] == dp[i + 1] and g[i] > g[i + 1]:
					g[i + 1] = g[i]
					
		return dp[n]