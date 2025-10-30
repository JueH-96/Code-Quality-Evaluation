from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        positive_nums = [num for num in nums if num > 0]
        negative_nums = [num for num in nums if num < 0]
        zero_count = nums.count(0)
        
        if len(nums) == 1:
            return nums[0]
        
        if len(negative_nums) % 2 != 0 and zero_count == 0:
            negative_nums.pop()
        
        product = 1
        for num in positive_nums + negative_nums:
            product *= num
        
        return product