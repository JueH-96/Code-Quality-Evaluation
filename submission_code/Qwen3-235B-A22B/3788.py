from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        unique_vals = set(nums)
        sum_positive = sum(x for x in unique_vals if x > 0)
        
        if sum_positive > 0:
            return sum_positive
        
        if 0 in unique_vals:
            return 0
        
        return max(unique_vals) if unique_vals else 0