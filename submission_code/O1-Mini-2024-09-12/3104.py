from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums_sorted = sorted(nums)
        res = 0
        
        # Check for s=0
        if n > 0 and nums_sorted[0] > 0:
            res +=1
        
        # Check for s from 1 to n-1
        for s in range(1, n):
            if nums_sorted[s-1] < s and nums_sorted[s] > s:
                res +=1
        
        # Check for s=n
        if n >0 and nums_sorted[-1] < n:
            res +=1
        
        return res