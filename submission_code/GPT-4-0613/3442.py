class Solution:
    def maxTotalReward(self, rewardValues):
        rewardValues.sort()
        x = 0
        for i in range(len(rewardValues)-1, -1, -1):
            if rewardValues[i] > x:
                x += rewardValues[i]
        return x