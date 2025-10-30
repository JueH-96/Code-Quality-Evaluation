from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        max_product = float('-inf')
        n = len(nums)
        
        # Iterate over all possible non-empty subsets
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if (mask >> i) & 1:
                    product *= nums[i]
            if product > max_product:
                max_product = product
        
        return max_product