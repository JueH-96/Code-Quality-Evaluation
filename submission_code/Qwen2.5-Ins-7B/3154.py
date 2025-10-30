from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_value = 0
        max_diff = 0
        current_max = nums[0]
        
        for j in range(1, len(nums) - 1):
            for k in range(j + 1, len(nums)):
                value = (current_max - nums[j]) * nums[k]
                max_value = max(max_value, value)
                max_diff = max(max_diff, current_max - nums[j])
                current_max = max(current_max, nums[j])
        
        return max(max_value, max_diff * nums[-1])