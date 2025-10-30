from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Distinct values that appear in the array
        unique_vals = set(nums)
        
        # The best we can do is to keep every positive distinct value
        positive_sum = sum(v for v in unique_vals if v > 0)
        
        if positive_sum > 0:
            return positive_sum
        
        # If there are no positive numbers, we must keep at least one element.
        # Choose the largest (closest to zero) number present.
        return max(nums)