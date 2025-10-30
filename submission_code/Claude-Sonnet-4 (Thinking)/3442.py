class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        
        # Use a set to track all achievable total rewards
        achievable = {0}
        
        for v in rewardValues:
            new_achievable = set()
            for current_sum in achievable:
                if v > current_sum:
                    new_achievable.add(current_sum + v)
            achievable.update(new_achievable)
        
        return max(achievable)