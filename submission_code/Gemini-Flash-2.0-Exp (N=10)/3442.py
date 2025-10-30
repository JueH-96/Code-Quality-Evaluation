class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues_with_indices = [(val, idx) for idx, val in enumerate(rewardValues)]
        rewardValues_with_indices.sort()
        
        total_reward = 0
        current_reward = 0
        marked_indices = set()
        
        for val, idx in rewardValues_with_indices:
            if val > current_reward:
                total_reward += val
                current_reward += val
                marked_indices.add(idx)
        
        return total_reward