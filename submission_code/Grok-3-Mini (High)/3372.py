from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0  # Handle empty array, though constraints ensure length >= 1
        
        max_length = 1
        curr_inc = 1
        curr_dec = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                curr_inc += 1
                curr_dec = 1
            elif nums[i] < nums[i - 1]:
                curr_dec += 1
                curr_inc = 1
            else:  # nums[i] == nums[i - 1]
                curr_inc = 1
                curr_dec = 1
            
            max_length = max(max_length, curr_inc, curr_dec)
        
        return max_length