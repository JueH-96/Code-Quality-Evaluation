from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        # Build prefix sum array p where p[i] = sum of nums[:i]
        p = [0] * (n + 1)
        for i in range(n):
            p[i+1] = p[i] + nums[i]
        
        total = 0
        # For each index i, compute start and add the subarray sum
        for i in range(n):
            start = max(0, i - nums[i])
            total += p[i+1] - p[start]
        
        return total