from typing import List
import math

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_product = -math.inf
        
        # iterate over each possible non-empty subset using bitmask
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if mask & (1 << i):
                    product *= nums[i]
            max_product = max(max_product, product)
        
        return max_product