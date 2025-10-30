from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        # Length of the array
        n = len(nums)
        if n == 0:
            return 0
        
        # Current lengths of increasing and decreasing contiguous subarrays
        inc_len = dec_len = 1
        best = 1
        
        for i in range(1, n):
            # Update increasing streak
            if nums[i] > nums[i - 1]:
                inc_len += 1
            else:
                inc_len = 1

            # Update decreasing streak
            if nums[i] < nums[i - 1]:
                dec_len += 1
            else:
                dec_len = 1

            best = max(best, inc_len, dec_len)
        
        return best