from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        sums = {0}
        for a in rewardValues:
            temp = set()
            for s in sums:
                if a > s:
                    temp.add(s + a)
            sums.update(temp)
        return max(sums)