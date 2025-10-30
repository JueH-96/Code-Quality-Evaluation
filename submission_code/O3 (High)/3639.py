from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        """
        For every position j we can decrement it at most once for every query
        that covers j.  Let cover[j] be the number of queries whose interval
        contains j.  The element nums[j] must not exceed cover[j]; if it does
        we can never perform enough decrements, otherwise we can simply apply
        exactly nums[j] of those cover[j] opportunities and skip the rest.

        So the task reduces to computing cover[j] for all indices and checking
        nums[j] ≤ cover[j].  We can obtain cover[] in O(n + q) with the usual
        range-addition / prefix-sum trick.
        """
        n = len(nums)
        diff = [0]*(n+1)           # difference array for range additions

        for l, r in queries:       # mark the ranges
            diff[l] += 1
            if r + 1 < n:
                diff[r+1] -= 1

        coverage = 0
        for i in range(n):         # build prefix sum and verify
            coverage += diff[i]
            if coverage < nums[i]:
                return False

        return True