class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        total_reward = 0
        marked = [False] * n
        
        while True:
            max_reward_index = -1
            max_reward = 0
            
            for i in range(n):
                if not marked[i] and rewardValues[i] > max_reward:
                    max_reward = rewardValues[i]
                    max_reward_index = i
            
            if max_reward_index == -1:
                break
            
            total_reward += max_reward
            marked[max_reward_index] = True
        
        return total_reward