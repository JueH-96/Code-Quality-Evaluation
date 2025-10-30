class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        groups = 0
        total = 0

        for i in range(len(usageLimits)):
            total += usageLimits[i]
            if total >= (groups + 1) * (groups + 2) // 2:
                groups += 1

        return groups