from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the reward values in descending order
        sorted_rewards = sorted(enumerate(rewardValues), key=lambda x: x[1], reverse=True)
        
        # Initialize the total reward and a set to keep track of marked indices
        total_reward = 0
        marked_indices = set()
        
        # Iterate over the sorted rewards
        for index, reward in sorted_rewards:
            # If the current index is not marked and the reward is greater than the total reward
            if index not in marked_indices and reward > total_reward:
                # Add the reward to the total reward and mark the index
                total_reward += reward
                marked_indices.add(index)
        
        # Return the maximum total reward
        return total_reward