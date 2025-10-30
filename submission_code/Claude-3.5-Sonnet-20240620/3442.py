class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        rewardValues.sort(reverse=True)
        total_reward = 0
        
        for reward in rewardValues:
            if reward > total_reward:
                total_reward += reward
            else:
                break
        
        return total_reward