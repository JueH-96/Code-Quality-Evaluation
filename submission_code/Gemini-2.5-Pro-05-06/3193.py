from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor_val = 0
        n = len(nums)
        
        # Iterate over all unique pairs of indices (i, j) such that i <= j.
        # This ensures that each pair of numbers (nums[i], nums[j]) is considered.
        # nums[i] is x, and nums[j] is y.
        for i in range(n):
            x = nums[i]
            # The inner loop starts from j = i. This covers:
            # 1. Pairs where the same element is chosen twice (when j == i).
            # 2. Pairs of distinct elements (when j > i), considering each such pair once.
            for j in range(i, n): 
                y = nums[j]
                
                # Check the strong pair condition: |x - y| <= min(x, y)
                # The condition is symmetric with respect to x and y.
                # abs(x - y) calculates the absolute difference.
                # min(x, y) correctly finds the smaller of the two values.
                if abs(x - y) <= min(x, y):
                    # If it's a strong pair, calculate its bitwise XOR
                    current_xor = x ^ y
                    # Update the overall maximum XOR found so far
                    if current_xor > max_xor_val:
                        max_xor_val = current_xor
                        
        return max_xor_val