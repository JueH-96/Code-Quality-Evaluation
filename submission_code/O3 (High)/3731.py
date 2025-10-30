from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Prefix sums: prefix[i] = sum(nums[0 .. i-1])
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        total = 0
        for i in range(n):
            start = max(0, i - nums[i])
            # sum of nums[start .. i] = prefix[i+1] - prefix[start]
            total += prefix[i + 1] - prefix[start]
        
        return total