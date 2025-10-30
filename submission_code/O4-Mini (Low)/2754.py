from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_strength = None
        
        # Enumerate all non-empty subsets via bitmasking
        for mask in range(1, 1 << n):
            prod = 1
            # Compute product for this subset
            for i in range(n):
                if mask & (1 << i):
                    prod *= nums[i]
            # Track maximum
            if max_strength is None or prod > max_strength:
                max_strength = prod
                
        return max_strength