class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Remove duplicates and sort the reward values
        unique_rewards = sorted(list(set(rewardValues)))
        max_reward = 0
        # Initialize a set to keep track of possible sums
        possible_sums = {0}
        
        for reward in unique_rewards:
            new_sums = set()
            for s in possible_sums:
                if reward > s:
                    new_sums.add(s + reward)
            possible_sums.update(new_sums)
            max_reward = max(max_reward, max(possible_sums, default=0))
        
        return max_reward