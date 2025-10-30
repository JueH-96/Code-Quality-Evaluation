class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        max_reward = max(rewardValues)
        max_x = 4000  # Since rewardValues[i] <= 2000, x cannot exceed 4000
        dp = [False] * (max_x + 1)
        dp[0] = True
        for val in rewardValues:
            for x in range(max_x - val, -1, -1):
                if dp[x] and val > x:
                    dp[x + val] = True
        for x in range(max_x, -1, -1):
            if dp[x]:
                return x
        return 0