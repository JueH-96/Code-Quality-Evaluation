class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Remove duplicates and sort the reward values
        unique_rewards = sorted(list(set(rewardValues)))
        # Initialize a set to keep track of possible sums
        possible_sums = {0}
        for reward in unique_rewards:
            # Create a new set to store updated sums
            new_sums = set()
            for s in possible_sums:
                if reward > s:
                    new_sums.add(s + reward)
            # Update the possible sums with the new sums
            possible_sums.update(new_sums)
        # The maximum sum is the maximum value in the possible sums set
        return max(possible_sums)