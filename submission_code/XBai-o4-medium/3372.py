from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 1
        
        prev_inc = 1
        prev_dec = 1
        max_len = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                current_inc = prev_inc + 1
                current_dec = 1
            elif nums[i] < nums[i-1]:
                current_inc = 1
                current_dec = prev_dec + 1
            else:
                current_inc = 1
                current_dec = 1
            max_len = max(max_len, current_inc, current_dec)
            prev_inc, prev_dec = current_inc, current_dec
        
        return max_len