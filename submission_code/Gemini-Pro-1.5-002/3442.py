class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        n = len(rewardValues)
        total_reward = 0
        for i in range(n):
            if rewardValues[i] > total_reward:
                total_reward += rewardValues[i]
        return total_reward