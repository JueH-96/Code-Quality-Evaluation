from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort(reverse=True)
        total_reward = 0
        current_reward = 0
        
        for value in rewardValues:
            if value > current_reward:
                current_reward += value
                total_reward += value
        
        return total_reward