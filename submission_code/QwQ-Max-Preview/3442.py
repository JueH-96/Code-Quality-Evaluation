class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        possible = {0}
        for r in rewardValues:
            new_sums = set()
            for s in possible:
                if s < r:
                    new_sums.add(s + r)
            possible.update(new_sums)
        return max(possible)