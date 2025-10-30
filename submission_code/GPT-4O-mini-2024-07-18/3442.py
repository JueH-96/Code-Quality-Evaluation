class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()  # Sort the reward values
        total_reward = 0
        current_reward = 0
        
        for value in rewardValues:
            if value > current_reward:
                total_reward += value
                current_reward += value
        
        return total_reward