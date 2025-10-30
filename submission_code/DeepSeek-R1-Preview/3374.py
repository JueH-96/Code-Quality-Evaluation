from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        total = 1
        prev_len = 1
        
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                current_len = prev_len + 1
            else:
                current_len = 1
            total += current_len
            prev_len = current_len
        
        return total