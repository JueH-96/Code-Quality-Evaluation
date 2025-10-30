from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        total_reward = 0
        
        for reward in rewardValues:
            if reward > total_reward:
                total_reward += reward
        
        return total_reward