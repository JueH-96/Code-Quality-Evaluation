class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        x = 0
        for reward in rewardValues:
            if reward > x:
                x += reward
        return x