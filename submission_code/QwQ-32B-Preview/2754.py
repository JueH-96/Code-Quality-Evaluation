from typing import List
import math

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        positives = [x for x in sorted_nums if x > 0]
        negatives = [x for x in sorted_nums if x < 0]
        
        # If number of negatives is odd, exclude the one with smallest absolute value
        if len(negatives) % 2 == 1:
            negatives = negatives[:-1]
        
        selected = positives + negatives
        
        # If selected is empty and zero is present, include zero
        if not selected:
            if 0 in nums:
                selected = [0]
            else:
                # Include the smallest negative if no zero is present
                selected = [sorted_nums[0]]
        
        product = math.prod(selected)
        return product