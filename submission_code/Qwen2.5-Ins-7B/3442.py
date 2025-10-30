class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        x = 0
        for value in rewardValues:
            if value > x:
                x += value
        return x