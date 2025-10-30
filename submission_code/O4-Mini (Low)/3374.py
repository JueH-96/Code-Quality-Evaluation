from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        total = 0
        curr_len = 1
        
        for i in range(1, n):
            if nums[i] != nums[i-1]:
                curr_len += 1
            else:
                # we finished an alternating segment of length curr_len
                total += curr_len * (curr_len + 1) // 2
                curr_len = 1
        
        # add the last segment
        total += curr_len * (curr_len + 1) // 2
        
        return total