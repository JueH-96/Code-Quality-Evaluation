class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        dp = {0}
        for e in rewardValues:
            next_dp = set()
            for s in dp:
                next_dp.add(s)  # option to not take e
                if e > s:
                    next_dp.add(s + e)  # option to take e
            dp = next_dp
        return max(dp)