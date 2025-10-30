from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        # At least every single element is a subarray of length 1
        best = 1
        inc_len = 1  # length of current strictly increasing run
        dec_len = 1  # length of current strictly decreasing run
        
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                inc_len += 1
            else:
                inc_len = 1
            
            if nums[i] < nums[i - 1]:
                dec_len += 1
            else:
                dec_len = 1
            
            best = max(best, inc_len, dec_len)
        
        return best