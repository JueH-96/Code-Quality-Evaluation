from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        start = 0
        
        for end in range(1, len(nums)):
            if nums[end] == nums[end - 1] + 1:
                if end - start == 1:
                    start = end - 1
                continue
            else:
                if (end - start) % 2 == 0 and nums[end] == nums[start]:
                    max_length = max(max_length, end - start + 1)
                start = end
        
        if (len(nums) - start) % 2 == 0 and nums[-1] == nums[start]:
            max_length = max(max_length, len(nums) - start)
        
        return max_length