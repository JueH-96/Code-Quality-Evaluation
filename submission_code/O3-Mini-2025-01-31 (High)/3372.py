from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0  # Given constraint ensures n >= 1, so this is just a safety check.
        
        max_len = 1
        inc = 1  # Length of current strictly increasing subarray
        dec = 1  # Length of current strictly decreasing subarray
        
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                inc += 1
                dec = 1  # reset strictly decreasing count
            elif nums[i] < nums[i - 1]:
                dec += 1
                inc = 1  # reset strictly increasing count
            else:
                # When equal, reset both counters since "strictly" increasing/decreasing is broken
                inc = 1
                dec = 1
            max_len = max(max_len, inc, dec)
        
        return max_len