class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        dp = [0]*n
        dp[0] = rewardValues[0]
        for i in range(1, n):
            dp[i] = max(dp[i-1], dp[i-2]+rewardValues[i])
        return dp[-1]