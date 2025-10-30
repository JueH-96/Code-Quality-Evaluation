from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the length of the longest sequential prefix
        current_len = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                current_len += 1
            else:
                break
        
        # Calculate the sum of the longest sequential prefix
        sum_seq = sum(nums[0:current_len])
        
        # Convert nums to a set for O(1) lookups
        set_nums = set(nums)
        
        # Find the smallest missing x >= sum_seq
        x = sum_seq
        while x in set_nums:
            x += 1
        
        return x