class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        rewardValues.sort()
        ans = mx = 0
        for i in range(n):
            if mx < rewardValues[i]:
                mx = rewardValues[i]
                ans += rewardValues[i]
        return ans