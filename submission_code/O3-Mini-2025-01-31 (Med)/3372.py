from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_length = 1
        inc = 1  # length of current strictly increasing subarray
        dec = 1  # length of current strictly decreasing subarray
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                inc += 1
                dec = 1  # reset decreasing streak
            elif nums[i] < nums[i - 1]:
                dec += 1
                inc = 1  # reset increasing streak
            else:
                # reset both if equality is found
                inc = 1
                dec = 1
            max_length = max(max_length, inc, dec)
        
        return max_length