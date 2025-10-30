class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        marked = [False] * n
        x = 0
        total_reward = 0
        
        sorted_indices = sorted(range(n), key=lambda i: rewardValues[i])
        
        for i in sorted_indices:
            if not marked[i] and rewardValues[i] > x:
                x += rewardValues[i]
                marked[i] = True
                total_reward += rewardValues[i]
                
        return total_reward