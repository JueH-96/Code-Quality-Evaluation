class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the reward values in ascending order
        rewardValues.sort()
        
        # Initialize the total reward x
        total_reward = 0
        
        # Iterate over the sorted reward values
        for reward in rewardValues:
            # If the current reward is greater than the total reward, add it
            if reward > total_reward:
                total_reward += reward
        
        return total_reward