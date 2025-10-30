class Solution:
	def minimumCoins(self, prices: List[int]) -> int:
		n = len(prices)
		dp = [0] * (n + 1)
		for i in range(1, n + 1):
			j_min = (i - 1) // 2
			best = float('inf')
			for j in range(j_min, i):
				total = dp[j] + prices[j]
				if total < best:
					best = total
			dp[i] = best
		return dp[n]