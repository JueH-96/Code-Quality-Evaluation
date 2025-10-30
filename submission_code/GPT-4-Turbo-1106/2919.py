class Solution:
    def maxIncreasingGroups(self, usageLimits: list[int]) -> int:
        usageLimits.sort()
        groups, size = 0, 0
        for limit in usageLimits:
            if limit > size:
                groups += 1
                size += 1
        return groups