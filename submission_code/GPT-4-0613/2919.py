class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        groups = 0
        for i in range(len(usageLimits) - 1, -1, -1):
            if usageLimits[i] > groups:
                groups += 1
        return groups