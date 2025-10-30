from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_len = 1
        prev_inc = 1
        prev_dec = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                current_inc = prev_inc + 1
                current_dec = 1
            elif nums[i] < nums[i-1]:
                current_dec = prev_dec + 1
                current_inc = 1
            else:
                current_inc = 1
                current_dec = 1
            current_max = max(current_inc, current_dec)
            if current_max > max_len:
                max_len = current_max
            prev_inc, prev_dec = current_inc, current_dec
        return max_len