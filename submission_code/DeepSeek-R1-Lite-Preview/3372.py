from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0  # Handle empty list, though constraints say 1 <= nums.length
        
        n = len(nums)
        current_increasing = 1
        current_decreasing = 1
        max_length = 1  # At least one element is always present
        
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                current_increasing += 1
                current_decreasing = 1
            elif nums[i] < nums[i - 1]:
                current_decreasing += 1
                current_increasing = 1
            else:
                current_increasing = 1
                current_decreasing = 1
            # Update max_length if necessary
            max_length = max(max_length, current_increasing, current_decreasing)
        
        return max_length