class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        total, groups = 0, 0
        for i in range(len(usageLimits)):
            total += usageLimits[i]
            if total >= usageLimits[i] * (groups + 1):
                groups += 1
            else:
                break
        return groups