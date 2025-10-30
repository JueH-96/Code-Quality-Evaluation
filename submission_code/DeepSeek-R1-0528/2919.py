class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        k = 0
        total = 0
        for x in usageLimits:
            total += x
            if total >= (k + 1) * (k + 2) // 2:
                k += 1
        return k