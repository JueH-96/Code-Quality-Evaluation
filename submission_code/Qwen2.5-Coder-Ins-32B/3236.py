from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the longest sequential prefix
        longest_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                longest_sum += nums[i]
            else:
                break
        
        # Find the smallest missing integer greater than or equal to longest_sum
        num_set = set(nums)
        x = longest_sum
        while x in num_set:
            x += 1
        
        return x