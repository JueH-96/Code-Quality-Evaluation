class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the reward values in ascending order
        rewardValues.sort()

        # Initialize the total reward and the current reward
        total_reward = 0
        current_reward = 0

        # Iterate through the sorted reward values
        for reward in rewardValues:
            # If the current reward is less than the reward value, add it to the total reward
            if current_reward < reward:
                total_reward += reward
                current_reward = reward

        return total_reward