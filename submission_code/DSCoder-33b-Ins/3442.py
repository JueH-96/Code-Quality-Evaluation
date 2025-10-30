class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        total_reward = 0
        while rewardValues:
            max_reward = rewardValues.pop()
            total_reward += max_reward
            if rewardValues:
                second_max_reward = rewardValues.pop(0)
                total_reward += second_max_reward
        return total_reward