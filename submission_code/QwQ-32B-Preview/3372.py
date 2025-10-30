from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0  # Edge case: empty list
        
        max_length = 1
        current_increasing = 1
        current_decreasing = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_increasing += 1
                current_decreasing = 1
                if current_increasing > max_length:
                    max_length = current_increasing
            elif nums[i] < nums[i - 1]:
                current_decreasing += 1
                current_increasing = 1
                if current_decreasing > max_length:
                    max_length = current_decreasing
            else:
                current_increasing = 1
                current_decreasing = 1
        
        return max_length