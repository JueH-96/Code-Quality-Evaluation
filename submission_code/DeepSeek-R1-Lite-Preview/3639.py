from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # Initialize difference array
        diff = [0] * (n + 1)
        
        # Apply queries using difference array
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n:
                diff[r + 1] -= 1
        
        # Compute prefix sum to get the count of queries covering each index
        c_i = 0
        for i in range(n):
            c_i += diff[i]
            if nums[i] > c_i:
                return False
        return True