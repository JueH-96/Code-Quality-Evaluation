class Solution:
	def maxTotalReward(self, rewardValues: List[int]) -> int:
		rewardValues.sort()
		n = len(rewardValues)
		dp = [0] * (n + 1)
		for i in range(1, n + 1):
			dp[i] = dp[i - 1]
			if rewardValues[i - 1] > 0:
				dp[i] = max(dp[i], rewardValues[i - 1])
			for j in range(i - 1):
				if rewardValues[j] < rewardValues[i - 1] and dp[j + 1] < rewardValues[i - 1]:
					if dp[j + 1] + rewardValues[i - 1] > dp[i]:
						dp[i] = dp[j + 1] + rewardValues[i - 1]
		return dp[n]