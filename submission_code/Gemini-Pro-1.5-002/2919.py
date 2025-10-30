class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        n = len(usageLimits)
        total_available = 0
        groups = 0
        for i in range(n):
            total_available += usageLimits[i]
            required = (groups + 1) * (groups + 2) // 2
            if total_available >= required:
                groups += 1
            else:
                break
        return groups