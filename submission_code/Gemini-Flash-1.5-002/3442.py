class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        rewardValues.sort()
        total_reward = 0
        current_reward = 0
        for i in range(n -1, -1, -1):
            if rewardValues[i] > current_reward:
                current_reward += rewardValues[i]
        return current_reward