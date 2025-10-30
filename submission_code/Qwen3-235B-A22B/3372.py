from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_len = 1
        incr = 1
        decr = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                incr += 1
                decr = 1
            elif nums[i] < nums[i-1]:
                decr += 1
                incr = 1
            else:
                incr = 1
                decr = 1
            
            max_len = max(max_len, incr, decr)
        
        return max_len