from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_length = 1
        inc = 1
        dec = 1
        for i in range(1, len(nums)):
            new_inc = 1
            new_dec = 1
            if nums[i] > nums[i-1]:
                new_inc = inc + 1
            if nums[i] < nums[i-1]:
                new_dec = dec + 1
            max_length = max(max_length, new_inc, new_dec)
            inc, dec = new_inc, new_dec
        return max_length