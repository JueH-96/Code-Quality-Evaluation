from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        # Initialize max_strength to negative infinity. This is important because
        # the maximum strength could be a negative number (e.g., if the input
        # array contains only negative numbers and the maximum product is negative)
        # or zero (e.g., if the input contains zeros and the best non-zero product
        # is negative). Any valid product from a non-empty subset will be greater
        # than negative infinity.
        max_strength = -float('inf') 

        # Iterate through all possible non-empty subsets of indices.
        # A bitmask 'mask' represents a subset of indices from 0 to n-1.
        # If the i-th bit is set in 'mask', it means the element nums[i] (at index i)
        # is included in the subset.
        # The loop runs from 1 up to (2^n) - 1 (inclusive).
        # mask = 0 corresponds to the empty set, which is excluded as the problem
        # requires a non-empty group.
        for mask in range(1, 1 << n):
            current_product = 1
            
            # Calculate the product for the subset represented by the current mask.
            # Iterate through each index 'i' from 0 to n-1.
            for i in range(n):
                # Check if the i-th bit is set in the current mask.
                # The expression (mask >> i) shifts the bit at position 'i' to the rightmost position.
                # The expression & 1 checks if the rightmost bit is 1.
                if (mask >> i) & 1:
                    # If the i-th bit is set, include the element nums[i] in the current product.
                    current_product *= nums[i]
            
            # After calculating the product for the current subset, update the maximum strength found so far.
            max_strength = max(max_strength, current_product)
            
        # After iterating through all non-empty subsets, max_strength holds the maximum product found.
        return max_strength