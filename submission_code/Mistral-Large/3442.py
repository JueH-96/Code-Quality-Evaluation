from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the reward values in ascending order
        rewardValues.sort()

        # Initialize the total reward
        total_reward = 0

        # Iterate through the sorted reward values
        for reward in rewardValues:
            # If the current reward is greater than the total reward, add it to the total reward
            if reward > total_reward:
                total_reward += reward

        # Return the maximum total reward
        return total_reward