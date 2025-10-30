from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        """
        Finds the maximum total reward that can be achieved by selecting unmarked indices
        where the reward is greater than the current total reward.
        
        Parameters:
        rewardValues (List[int]): List of reward values for each index.
        
        Returns:
        int: Maximum total reward achievable.
        """
        sorted_indices = sorted(range(len(rewardValues)), key=lambda i: rewardValues[i], reverse=True)
        max_reward = 0
        total_reward = 0
        for index in sorted_indices:
            if rewardValues[index] > total_reward:
                total_reward += rewardValues[index]
                max_reward = max(max_reward, total_reward)
        
        return max_reward