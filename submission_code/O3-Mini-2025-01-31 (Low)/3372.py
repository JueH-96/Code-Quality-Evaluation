from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        # Edge case: if list is empty, return 0. However by constraint, length >=1.
        if not nums:
            return 0
        
        max_len = 1
        inc = 1   # length of current strictly increasing subarray
        dec = 1   # length of current strictly decreasing subarray
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                inc += 1
            else:
                inc = 1  # reset the increasing counter
            
            if nums[i] < nums[i - 1]:
                dec += 1
            else:
                dec = 1  # reset the decreasing counter
            
            max_len = max(max_len, inc, dec)
        
        return max_len