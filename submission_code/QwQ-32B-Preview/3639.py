from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        
        for q in queries:
            l, r = q
            diff[l] += 1
            if r + 1 < n:
                diff[r + 1] -= 1
        
        count = 0
        for i in range(n):
            count += diff[i]
            if count < nums[i]:
                return False
        return True