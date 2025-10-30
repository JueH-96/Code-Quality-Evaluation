from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_value = 0
        max_diff = 0
        max_num = nums[0]
        
        for j in range(1, len(nums) - 1):
            max_diff = max(max_diff, max_num - nums[j])
            for k in range(j + 1, len(nums)):
                max_value = max(max_value, max_diff * nums[k])
            max_num = max(max_num, nums[j])
        
        return max_value