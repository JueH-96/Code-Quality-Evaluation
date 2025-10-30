import math

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Initialize max_strength to negative infinity.
        # This handles cases where all possible products are negative (e.g., nums = [-5]),
        # ensuring that the largest (least negative) product is correctly captured.
        max_strength = -float('inf') 
        
        # Iterate through all possible non-empty subsets using bitmasks.
        # A bitmask 'i' represents a subset where the j-th bit of 'i' being set
        # means nums[j] is included in the subset.
        # The loop starts from 1 (binary 0...01) to ensure non-empty subsets.
        # It goes up to (1 << n) - 1 (binary 1...11) to cover all subsets.
        for i in range(1, 1 << n):
            current_product = 1
            
            # Iterate through each element in nums to check if it's part of the current subset.
            for j in range(n):
                # If the j-th bit of the mask 'i' is 1, include nums[j] in the product.
                if (i >> j) & 1:
                    current_product *= nums[j]
            
            # Update the overall maximum strength found so far.
            max_strength = max(max_strength, current_product)
            
        return max_strength