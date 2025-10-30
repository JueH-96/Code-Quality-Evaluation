class Solution:
	def maxTotalReward(self, rewardValues: List[int]) -> int:
		total_sum = sum(rewardValues)
		dp = [False] * (total_sum + 1)
		dp[0] = True
		max_total = 0
		arr = sorted(rewardValues)
		for r in arr:
			end = min(max_total, r - 1)
			for x in range(end, -1, -1):
				if dp[x]:
					new_total = x + r
					dp[new_total] = True
					if new_total > max_total:
						max_total = new_total
		return max_total