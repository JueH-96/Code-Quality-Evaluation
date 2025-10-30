class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        n = len(usageLimits)
        total = 0
        res = 0
        j = 0
        for i in range(n-1, -1, -1):
            total += usageLimits[i]
            j += 1
            if total >= j * (j + 1) // 2:
                res = j
        return res