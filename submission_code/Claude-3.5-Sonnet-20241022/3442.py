class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        # Sort indices by reward values
        indices = sorted(range(n), key=lambda x: rewardValues[x])
        
        total_reward = 0
        used = [False] * n
        
        # Try each index in ascending order of reward values
        for i in range(n):
            curr_idx = indices[i]
            # If current reward value is greater than total reward so far
            if rewardValues[curr_idx] > total_reward:
                total_reward += rewardValues[curr_idx]
                used[curr_idx] = True
        
        return total_reward