class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        sum_so_far = 0
        k = 0
        for u in usageLimits:
            sum_so_far += u
            if sum_so_far >= (k + 1) * (k + 2) // 2:
                k += 1
        return k