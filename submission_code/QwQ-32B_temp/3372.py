from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        
        max_len = 1
        inc = 1
        dec = 1
        
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                new_inc = inc + 1
                new_dec = 1
            elif nums[i] < nums[i-1]:
                new_dec = dec + 1
                new_inc = 1
            else:
                new_inc = 1
                new_dec = 1
            
            inc, dec = new_inc, new_dec
            current_max = max(inc, dec)
            if current_max > max_len:
                max_len = current_max
        
        return max_len