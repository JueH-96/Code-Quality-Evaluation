from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        # Build prefix sum array where prefix[i] is sum of nums[0..i-1]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        total = 0
        for i, v in enumerate(nums):
            start = max(0, i - v)
            # sum of nums[start..i] is prefix[i+1] - prefix[start]
            total += prefix[i+1] - prefix[start]
        
        return total