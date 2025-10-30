class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        achievable = {0}
        for reward in rewardValues:
            new_achievable = set()
            for x in achievable:
                if reward > x:
                    new_achievable.add(x + reward)
            achievable.update(new_achievable)
        return max(achievable)